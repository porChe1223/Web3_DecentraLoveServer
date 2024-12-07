from typing import Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  #リクエストbodyを定義するために必要
from llama_cpp import Llama  #Llama3モデルに必要
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, PreTrainedTokenizerFast, Seq2SeqTrainingArguments, AutoModelForCausalLM
from datasets import load_dataset
import torch
from transformers import pipeline, AutoTokenizer
from dotenv import load_dotenv
import os
import zipfile
import shutil
import httpx
import aiofiles
import requests
from pathlib import Path
import tarfile

# パーツを保存するディレクトリ
PARTS_DIRECTORY = './parts'
os.makedirs(PARTS_DIRECTORY, exist_ok=True)

qwen_model_name = 'Qwen2-0.5B-Instruct'
gemma_model_name = 'gemma-2-baku-2b-it'
llama_1B_model_name = ['Llama-3.2-1B-Instruct', 'ojosama-llama-1B']
llama_gguf_model_name = ['Llama-2-7b-Japanese', 'haqishen-Llama-3-8B-Japanese', 'umiyuki-Japanese-Chat-Umievo', 'lightblue-suzume-llama-3-8B-japanese']

model_name_to_huggingface_name = {
    'Qwen2-0.5B-Instruct': 'Qwen/Qwen2-0.5B-Instruct',
    'gemma-2-baku-2b-it': 'rinna/gemma-2-baku-2b-it',
    'Llama-3.2-1B-Instruct': 'meta-llama/Llama-3.2-1B-Instruct',
    'ojosama-llama-1B': './app/results/ojosama-llama-1B'
}
gguf_model_name_to_model = {
    "Llama-2-7b-Japanese": {
        "model_repo_id": "mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf",
        "model_filename": "ELYZA-japanese-Llama-2-7b-fast-instruct-q2_K.gguf"
    },
    "haqishen-Llama-3-8B-Japanese": {
        "model_repo_id": "mmnga/haqishen-Llama-3-8B-Japanese-Instruct-gguf",
        "model_filename": "haqishen-Llama-3-8B-Japanese-Instruct-Q3_K_L.gguf"
    },
    "umiyuki-Japanese-Chat-Umievo": {
        "model_repo_id": "mmnga/umiyuki-Japanese-Chat-Umievo-itr001-7b-gguf",
        "model_filename": "umiyuki-Japanese-Chat-Umievo-itr001-7b-Q2_K.gguf"
    },
    "lightblue-suzume-llama-3-8B-japanese": {
        "model_repo_id": "mmnga/lightblue-suzume-llama-3-8B-japanese-gguf",
        "model_filename": "lightblue-suzume-llama-3-8B-japanese-Q4_K_M.gguf"
    }
}
load_dotenv()
token = os.getenv('TOKEN')
url = os.getenv("DECENTRA_URL")
wallet = os.getenv("DECENTRA_WALLET")
model = os.getenv("DECENTRA_MODEL")
# host_url = "http://192.168.11.5:8082"
host_url = os.getenv("DECENTRA_HOST_URL")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Environment => url: {url}, wallet: {wallet}, model: {model}")
    folder_path = Path("./app/results/ojosama-llama-1B")
    exists = folder_path.is_dir()
    if (model == "ojosama-llama-1B" and not exists):
        print("Downloading ojosama Model...")
        response = requests.get(f"{host_url}/files/models_params/{model}")
        saveFilePath = f"./app/results/cache.tar"
        with open(saveFilePath, 'wb') as saveFile:
            saveFile.write(response.content)
        with tarfile.open(saveFilePath) as folder:
            folder.extractall(path="./app/results")    

    json_params = {
        "IPAddress": url,
        "Model_Name": model,
        "Wallet_Address": wallet
    }
    res = requests.post(f"{host_url}/register_GPUProvider", json=json_params)
    if (res.status_code == 200):
        print("setup GPU_Provider Success!!")
    else:
        print("setup GPU_Provider Failed")
        os._exit(1)
    yield
    json_params = {
        "ip_address": url,
    }
    res = requests.delete(f"{host_url}/gpu_providers", json=json_params)
    print(res.status_code)
    if (res.status_code == 200):
        print("Deleted GPU_Provider")
    else:
        print("Delete GPU_Provider Failed")
    print("shutdown!!")
    

app = FastAPI(lifespan=lifespan)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  #セキュリティに注意
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

################
### 推論API ####
################
class InferenceModel(BaseModel): # 推論用のモデルの選択
    model_name: str

class Question(BaseModel): # 質問文の選択
    content: str

class SavePartRequest(BaseModel):
    name_content_hash: str
    part_index: int

class SaveFailureRequest(BaseModel):
    name_content_hash: str

def get_save_part_request(
    name_content_hash: str = Query(..., description="名前とコンテンツのハッシュ"),
    part_index: int = Query(..., description="パートのインデックス")
) -> SavePartRequest:
    return SavePartRequest(name_content_hash=name_content_hash, part_index=part_index)

def get_save_failure_request(
    name_content_hash: str = Query(..., description="名前とコンテンツのハッシュ")
) -> SaveFailureRequest:
    return SaveFailureRequest(name_content_hash=name_content_hash)

# 接続確認
@app.get('/check_connect')
def check_connect():
    """
    接続確認用のエンドポイント。
    GETリクエストに対して、ステータス200とシンプルなメッセージを返します。
    """
    return {
        "status": "OK"
    }

# パーツの取得、保存

@app.get("/get_part")
async def get_part(
    name_content_hash: str = Query(..., description="名前とコンテンツのハッシュ"),
    part_index: int = Query(..., description="パートのインデックス")
):
    """
    指定されたパーツファイルを返すエンドポイント。
    """
    part_filename = f"{name_content_hash}_part_{part_index}"
    part_path = f"{PARTS_DIRECTORY}/{part_filename}"

    return FileResponse(
        path=part_path,
        media_type='application/octet-stream',
        filename=part_filename
    )

@app.post("/save_part")
async def save_part(
    save_part_request: SavePartRequest = Depends(get_save_part_request),
    file: UploadFile = File(...)
):
    """
    パーツファイルを保存するエンドポイント。
    """
    part_filename = f"{save_part_request.name_content_hash}_part_{save_part_request.part_index}"
    part_path = f"{PARTS_DIRECTORY}/{part_filename}"

    try:
        with open(part_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    return {"message": "Part saved successfully"}

@app.post("/save_failure")
async def save_failure(
    save_failure_request: SaveFailureRequest = Depends(get_save_failure_request)
):
    """
    失敗通知を処理するエンドポイント。
    """
    # 失敗通知を処理するロジックをここに追加
    print(f"Failed to save file with hash: {save_failure_request.name_content_hash}")
    return {"message": "Failure notified"}

# エラーハンドリングの例（オプション）
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

# 推論
@app.post("/")
def create_answer(question: Question, model: InferenceModel):
    model_name = model.model_name
    if (model_name in llama_gguf_model_name):
        llm = Llama.from_pretrained(
            repo_id=gguf_model_name_to_model[model_name]['model_repo_id'],
            filename=gguf_model_name_to_model[model_name]['model_filename']
        )

        answer = llm.create_chat_completion(
            messages = [
                {
                    "role": "user",
                    "content": question.content  #Webページ側で受け取るメッセージ
                }
            ],
            max_tokens = 300
        )
        return {
            "answer": answer['choices'][0]['message']['content']
        }

    if (model_name in llama_1B_model_name):
        ft_model = AutoModelForCausalLM.from_pretrained(
            model_name_to_huggingface_name[model_name],
            device_map='auto',
            torch_dtype=torch.bfloat16,
            token=token
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_to_huggingface_name[model_name], token=token)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        messages = [
            {"role": "system", "content": "あなたは日本語で回答するアシスタントです。"},
            {"role": "user", "content": question.content},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(ft_model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = ft_model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1]:]

        return {
            "answer": tokenizer.decode(response, skip_special_tokens=True)
        }

    if (model_name == qwen_model_name):
        return chat_qwen(question, modelName=model_name_to_huggingface_name[model_name])

    if (model_name == gemma_model_name):
        return chat_gemma(question, modelName=model_name_to_huggingface_name[model_name])

    raise HTTPException(status_code=404, detail="Item not found")


# Qwen公式コードから抜粋
def chat_qwen(question: Question, modelName: str):
    ft_model = AutoModelForCausalLM.from_pretrained(
        modelName,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(modelName, token=token)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    messages = [
        {"role": "system", "content": "あなたは日本語で回答するアシスタントです。"},
        {"role": "user", "content": question.content},
    ]

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(ft_model.device)

    generated_ids = ft_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return {"answer": response}

# Gemma 2 Baku Instruction
def chat_gemma(question: Question, modelName: str):
    ft_model = AutoModelForCausalLM.from_pretrained(
        modelName,
        # device_map='auto',
        # torch_dtype=torch.bfloat16,
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(modelName, token=token)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    messages = [
        # System Role使えない
        {"role": "user", "content": question.content},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(ft_model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = ft_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]

    return {
        "answer": tokenizer.decode(response, skip_special_tokens=True)
    }

################
### 学習API ####
################
class TrainModel(BaseModel): # 学習用モデル
    model_id: str = "Amu/supertiny-llama3-0.25B-v0.1",

class Tokenizer(BaseModel): # トークナイザー
    tokenizer_id: str = "Amu/supertiny-llama3-0.25B-v0.1",
    #tokenizer.pad_token = tokenizer.eos_token

class UserToken(BaseModel): #ユーザートークン
    user_token: str = os.getenv('TOKEN')

class Dataset(BaseModel): # データセット
    dataset: str = 'xsum'
    #small_dataset: dataset.shuffle(seed=42).select([i for i in range(1000)])

# 学習詳細設定
class TrainRequest(BaseModel):
    learning_rate: float = 2e-5           # 学習率
    batch_size: int = 1                  # バッチサイズ
    gradient_accumulation_steps: int = 4  # 勾配の累積数
    epochs: int = 3                       # エポック数
    weight_decay: float = 0.01            # 重み減衰
    output_dir: str = "./result"          # 保存先


# 学習（ファインチューニング）
@app.post("/train")
async def train_model(model:TrainModel, tokenizer:Tokenizer, user_token:UserToken, dataset:Dataset, request:TrainRequest):
    # モデルの読み込み
    print("モデルを読み込み中...")
    model = LlamaForCausalLM.from_pretrained(
        model.model_id,
        token = user_token.user_token
    )
    print("モデル読み込み完了")

    #トークナイザーの読み込み
    print("トークナイザーを読み込み中...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer.tokenizer_id,
        token = user_token.user_token
        #legacy=False  # 新しい動作を有効にする
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("トークナイザー読み込み完了")

    # データセットの準備
    print("データセットを読み込み中...")
    dataset = load_dataset(dataset.dataset, split="train", trust_remote_code=True)
    print("データセット読み込み完了")


    #トークナイザーの設定
    def preprocess_data(examples):
        inputs = examples["document"]
        outputs = examples["summary"]
        model_inputs = tokenizer(inputs, text_target=outputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    #トークナイズされたデータセットの取得
    print("データセットをトークナイズ中...")
    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    print("トークナイズ完了")

    # トレーニングデータと評価データに分割
    train_size = 0.8
    train_dataset = tokenized_dataset.shuffle(seed=42).select(range(int(len(tokenized_dataset) * train_size)))
    eval_dataset = tokenized_dataset.select(range(int(len(tokenized_dataset) * train_size), len(tokenized_dataset)))

    # トレーニングの設定
    print("トレーニング設定を構築中...")
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy = "epoch",
        fp16 = True,
        learning_rate = request.learning_rate,
        per_device_train_batch_size = request.batch_size,
        per_device_eval_batch_size = request.batch_size,
        gradient_accumulation_steps = request.gradient_accumulation_steps,
        num_train_epochs = request.epochs,
        weight_decay = request.weight_decay,
        output_dir = request.output_dir
    )
    print("トレーニング設定完了")

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_dataset,
        eval_dataset=eval_dataset
    )
    
    # ファインチューニングの実行
    print("トレーニング開始...")
    trainer.train()
    print("トレーニング完了")
    
    # モデルの保存
    print("モデル保存中...")
    trainer.save_model(request.output_dir)
    print("モデル保存完了")

    # Zipファイル化
    print("Zipファイル化中...")
    shutil.make_archive('result_zip', format='zip', root_dir=request.output_dir)
    print("Zipファイル化完了")
    
    #管理サーバに送り返す
    async with aiofiles.open('./result_zip', 'rb') as f:  #Zipファイルの中身を取得
        part_content = await f.read()

    async with httpx.AsyncClient() as client:
        response = client.post(
            'http://192.168.11.17:8000/',
            files={"file": (f"{result_zip}", part_content)}
        )


    return {"status": "Training completed", "epochs": request.epochs, "batch_size": request.batch_size}