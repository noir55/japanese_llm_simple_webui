#!/bin/env python3

import argparse
import gradio as gr
import torch
import time
import numpy as np
from torch.nn import functional as F
import os
import re
from threading import Thread
from peft import PeftModel
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


#------
# 設定
#------

# バージョン
VERSION = "1.0.0"

# ページの最上部に表示させたいタイトルを設定
TITLE_STRINGS = "Rinna 3.6B Instruction PPO Chat"

# モデルタイプ("rinna","opencalm","llama")
MODEL_TYPE = "rinna"
# ベースモデルを設定
BASE_MODEL = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
# トークナイザ―の設定
TOKENIZER_MODEL = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
# モデルを8ビット量子化で実行するか("on","off")
LOAD_IN_8BIT = "off"

# LoRAのディレクトリ(空文字列に設定すると読み込まない)
LORA_WEIGHTS = ""

# プロンプトタイプ("rinna","vicuna","alpaca","none")
PROMPT_TYPE = "rinna"
# プロンプトが何トークンを超えたら履歴を削除するか
PROMPT_THRESHOLD = 1024
# 履歴を削除する場合、何トークン未満まで削除するか
PROMPT_DELETED = 512

# 推論時に生成する最大トークン数
MAX_NEW_TOKENS = 512
# 推論時の出力の多様さ(大きいほどバリエーションが多様になる)
TEMPERATURE = 0.7

# WebUIがバインドするIPアドレス
GRADIO_HOST = '127.0.0.1'
# WebUIがバインドするポート番号
GRADIO_PORT = 7860

# デバッグメッセージを標準出力に表示するか("on","off")
DEBUG_FLAG = "on"


#------------------
# クラス、関数定義
#------------------

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def user(message, history):
    # Rinnaモデルの場合"<NL>"を改行に変換
    if MODEL_TYPE == "rinna":
        for item in history:
            item[0] = re.sub("<NL>", "\n", item[0])
            item[1] = re.sub("<NL>", "\n", item[1])
    # <br>が増殖するのを防止
    for item in history:
        item[0] = re.sub("<br>\n", "\n", item[0])
        item[1] = re.sub("<br>\n", "\n", item[1])
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]

# Regenerateボタンクリック時の動作
def regen(history):
    if len(history) == 0:
        return "", [["", ""]]
    else:
        history[-1][1]=""
        # Rinnaモデルの場合"<NL>"を改行に変換
        if MODEL_TYPE == "rinna":
            for item in history:
                item[0] = re.sub("<NL>", "\n", item[0])
                item[1] = re.sub("<NL>", "\n", item[1])
        # <br>が増殖するのを防止
        for item in history:
            item[0] = re.sub("<br>\n", "\n", item[0])
            item[1] = re.sub("<br>\n", "\n", item[1])
        return history[-1][0], history

# Remove lastボタンクリック時の動作
def remove_last(history):
    if len(history) == 0:
        return "", [["", ""]]
    else:
        history.pop(-1)
        # <br>が増殖するのを防止
        for item in history:
            item[0] = re.sub("<br>\n", "\n", item[0])
            item[1] = re.sub("<br>\n", "\n", item[1])
        return history

# プロンプト文字列を生成する関数
def prompt(curr_system_message, history):

    # Rinna-3.6B形式のプロンプト生成
    if PROMPT_TYPE == "rinna":
        messages = curr_system_message + \
            new_line.join([new_line.join(["ユーザー: "+item[0], "システム: "+item[1]])
                    for item in history])
    # Vicuna形式のプロンプト生成
    elif PROMPT_TYPE == "vicuna":
        prefix = f"""A chat between a curious user and an artificial intelligence assistant.{new_line}The assistant gives helpful, detailed, and polite answers to the user's questions.{new_line}{new_line}"""
        messages = curr_system_message + \
            new_line.join([new_line.join(["USER: "+item[0], "ASSISTANT: "+item[1]])
                    for item in history])
        messages = prefix + messages
    # Alpaca形式のプロンプト生成
    elif PROMPT_TYPE == "alpaca":
        prefix = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.{new_line}{new_line}"""
        messages = curr_system_message + \
            f"{new_line}{new_line}".join([new_line.join([f"### Instruction:{new_line}"+item[0], f"{new_line}### Response:{new_line}"+item[1]])
                    for item in history])
        messages = prefix + messages
    # 特定の書式を使用しない(入力した文章の続きを生成する)場合のプロンプト生成
    elif PROMPT_TYPE == "none":
        messages = curr_system_message + \
            "".join(["".join([item[0], item[1]])
                    for item in history])
    # PROMPT_TYPE設定が正しくなければ終了する
    else:
        print(f"Invalid PROMPT_TYPE \"{PROMPT_TYPE}\"")
        exit()
    # 生成したプロンプト文字列を返す
    return messages


def chat(curr_system_message, history):

    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    # "<br>"を削除しておく(モデルに付加された<br>タグが渡らないようにする)
    for item in history:
        item[0] = re.sub("<br>\n", "\n", item[0])
        item[1] = re.sub("<br>\n", "\n", item[1])

    # 会話履歴を表示
    if DEBUG_FLAG:
        print(f"history={history}\n")

    # プロンプト文字列生成
    del_flag = 0
    while True:
        # プロンプト文字列を生成する
        messages = prompt(curr_system_message, history)
        # プロンプトをトークナイザで変換する
        if MODEL_TYPE == "rinna":
            messages = re.sub("\n", "<NL>", messages)
            model_inputs = tok([messages], return_tensors="pt", add_special_tokens=False, padding=True)
        elif MODEL_TYPE == "opencalm":
            model_inputs = tok([messages], return_tensors="pt")
        elif MODEL_TYPE == "llama":
            model_inputs = tok([messages], return_tensors="pt")
        # もしプロンプトのトークン数が多すぎる場合は削除フラグを設定
        if del_flag == 0 and len(model_inputs['input_ids'][0]) > PROMPT_THRESHOLD:
            del_flag = 1
        # 削除フラグが設定され、かつPROMPT_DELETEDよりトークン数が多い場合は履歴の先頭を削除
        if del_flag == 1 and len(model_inputs['input_ids'][0]) > PROMPT_DELETED:
            history.pop(0)
        # 削除フラグが設定されてないか、設定されているがPROMPT_DELETEDよりトークン数が少ない場合ループを抜ける
        else:
            break

    # プロンプトを標準出力に表示
    if DEBUG_FLAG:
        print(f"--prompt strings--\n{messages}\n----\n")
        print(f"--prompt tokens--\n{model_inputs}\n----\n")

    # 入力トークンをGPUに送る
    model_inputs = model_inputs.to("cuda")

    # モデルに入力して回答を生成(ストリーミング出力させる)
    streamer = TextIteratorStreamer(
        tok, timeout=60., skip_prompt=True, skip_special_tokens=True)

    # 推論設定
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        #top_p=0.95,
        #top_k=40,
        temperature=TEMPERATURE,
        num_beams=1,
        repetition_penalty=1.1,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=m.generate, kwargs=generate_kwargs)

    # スレッドで生成を実行
    t.start()

    #print(history)
    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        # Rinnaモデルの場合"<NL>"を改行に変換
        if MODEL_TYPE == "rinna":
            new_text = re.sub("<NL>", "\n", new_text)
        # print(new_text)
        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to cleanup the message textbox and the updated conversation history
        yield history
    return partial_text


#------
# 実行
#------

# 引数を取得
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=BASE_MODEL, help="モデル名またはディレクトリのパス")
parser.add_argument("--model-type", type=str, choices=["rinna", "opencalm", "llama"],  default=MODEL_TYPE, help="モデルタイプ名")
parser.add_argument("--tokenizer", type=str, default=TOKENIZER_MODEL, help="トークナイザー名またはディレクトリのパス")
parser.add_argument("--load-in-8bit", type=str, choices=["on", "off"], default=LOAD_IN_8BIT, help="モデル名またはディレクトリのパス")
parser.add_argument("--lora", type=str, default=LORA_WEIGHTS, help="LoRAディレクトリのパス")
parser.add_argument("--prompt-type", type=str, choices=["rinna", "vicuna", "alpaca", "none"], default=PROMPT_TYPE, help="プロンプトタイプ名")
parser.add_argument("--prompt-threshold", type=int, default=PROMPT_THRESHOLD, help="このトークン数を超えたら古い履歴を削除")
parser.add_argument("--prompt-deleted", type=int, default=PROMPT_DELETED, help="古い履歴削除時にこのトークン以下にする")
parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="推論時に生成するトークン数の最大")
parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="生成する文章の多様さ")
parser.add_argument("--host", type=str, default=GRADIO_HOST, help="WebサーバがバインドするIPアドレスorホスト名")
parser.add_argument("--port", type=int, default=GRADIO_PORT, help="Webサーバがバインドするポート番号")
parser.add_argument("--title", type=str, default=TITLE_STRINGS, help="Webページのタイトル")
parser.add_argument("--debug", type=str, choices=["on", "off"], default=DEBUG_FLAG, help="デバッグメッセージを標準出力に表示")
args = parser.parse_args()

# 引数でセットされた値で上書きする
BASE_MODEL = args.model
MODEL_TYPE = args.model_type
TOKENIZER_MODEL = args.tokenizer
LOAD_IN_8BIT = args.load_in_8bit
LORA_WEIGHTS = args.lora
PROMPT_TYPE = args.prompt_type
PROMPT_THRESHOLD = args.prompt_threshold
PROMPT_DELETED = args.prompt_deleted
MAX_NEW_TOKENS = args.max_new_tokens
TEMPERATURE = args.temperature
GRADIO_HOST = args.host
GRADIO_PORT = args.port
TITLE_STRINGS = args.title
DEBUG_FLAG = args.debug

# パラメータ表示
print("---- パラメータ ----")
print(f"モデル名orパス: {BASE_MODEL}")
print(f"モデルタイプ名: {MODEL_TYPE}")
print(f"トークナイザー: {TOKENIZER_MODEL}")
print(f"8bit量子化: {LOAD_IN_8BIT}")
if LORA_WEIGHTS == "":
    print(f"LoRAモデルパス: (LoRAなし)")
else:
    print(f"LoRAモデルパス: {LORA_WEIGHTS}")
print(f"プロンプトタイプ: {PROMPT_TYPE}")
print(f"プロンプトトークン数しきい値: {PROMPT_THRESHOLD}")
print(f"プロンプトトークン数削除値: {PROMPT_DELETED}")
print(f"生成最大トークン数: {MAX_NEW_TOKENS}")
print(f"Temperature: {TEMPERATURE}")
print(f"WebサーバIPorホスト名: {GRADIO_HOST}")
print(f"Webサーバポート番号: {GRADIO_PORT}")
print(f"Webページタイトル: {TITLE_STRINGS}")
print(f"デバッグ: {DEBUG_FLAG}\n")

# LOAD_IN_8BITはTrue or Falseに変換
if LOAD_IN_8BIT == "on":
    LOAD_IN_8BIT = True
else:
    LOAD_IN_8BIT = False
# DEBUG_FLAGはTrue or Falseに変換
if DEBUG_FLAG == "on":
    DEBUG_FLAG = True
else:
    DEBUG_FLAG = False

## モデルタイプによる設定とモデルのロード

# Rinna-3.6Bモデルの場合
if MODEL_TYPE == "rinna":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # 改行を示す文字の設定
    new_line = "<NL>"
    # モデルのロード
    print(f"Starting to load the model \"{BASE_MODEL}\" to memory")
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float16, 
        load_in_8bit=LOAD_IN_8BIT, 
        device_map='auto'
        )
    print(f"Sucessfully loaded the model to the memory")
    # トークナイザ―のロード
    print(f"Starting to load the tokenizer \"{TOKENIZER_MODEL}\" to memory")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=False)
    print(f"Sucessfully loaded the tokenizer to the memory")
    # padding設定
    m.config.pad_token_id = tok.eos_token_id
# Open CALMモデルの場合
elif MODEL_TYPE == "opencalm":
    from transformers import AutoModelForCausalLM, GPTNeoXTokenizerFast
    # 改行を示す文字の設定
    new_line = "\n"
    # モデルのロード
    print(f"Starting to load the model \"{BASE_MODEL}\" to memory")
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float16, 
        load_in_8bit=LOAD_IN_8BIT, 
        device_map='auto'
        )
    print(f"Sucessfully loaded the model to the memory")
    # トークナイザ―のロード
    print(f"Starting to load the tokenizer \"{TOKENIZER_MODEL}\" to memory")
    tok = GPTNeoXTokenizerFast.from_pretrained(TOKENIZER_MODEL)
    print(f"Sucessfully loaded the tokenizer to the memory")
# Llama系モデルの場合
elif MODEL_TYPE == "llama":
    from transformers import LlamaForCausalLM, LlamaTokenizer
    # 改行を示す文字の設定
    new_line = "\n"
    # モデルのロード
    print(f"Starting to load the model \"{BASE_MODEL}\" to memory")
    m = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float16, 
        load_in_8bit=LOAD_IN_8BIT, 
        device_map='auto'
        )
    print(f"Sucessfully loaded the model to the memory")
    # トークナイザ―のロード
    print(f"Starting to load the tokenizer \"{TOKENIZER_MODEL}\" to memory")
    tok = LlamaTokenizer.from_pretrained(TOKENIZER_MODEL)
    print(f"Sucessfully loaded the tokenizer to the memory")
# MODEL_TYPE設定が正しくなければ終了する
else:
    print(f"Invalid MODEL_TYPE \"{MODEL_TYPE}\"")
    exit()

# ジェネレータの作成 (不要だと思われるためコメントアウト)
#generator = pipeline('text-generation', model=m, tokenizer=tok)

# LoRAのロード
if LORA_WEIGHTS != "":
    print(f"Starting to load the LoRA weights \"{LORA_WEIGHTS}\" to memory")
    m = PeftModel.from_pretrained(m, LORA_WEIGHTS, torch_dtype=torch.float16)
    print(f"Sucessfully loaded the LoRA weights to the memory")

# プロンプトの先頭に付加する文字列
start_message = ""


# Webページ
with gr.Blocks(title="LLM Simple WebUI", theme=gr.themes.Base()) as demo:
    history = gr.State([])
    gr.Markdown(f"## {TITLE_STRINGS}")
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        with gr.Column(scale=20):
            msg = gr.Textbox(label="Chat Message Box", placeholder="Chat Message Box",
                             show_label=False).style(container=False)
        with gr.Column(scale=1, min_width=100):
            submit = gr.Button("Submit")
    with gr.Row():
                stop = gr.Button("Stop")
                regenerate = gr.Button("Regenerate")
                removelast = gr.Button("Remove last")
                clear = gr.Button("Clear")
    system_msg = gr.Textbox(
        start_message, label="System Message", interactive=False, visible=False)

    submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True)
    submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[
               submit_event, submit_click_event], queue=False)
    regenerate.click(fn=regen, inputs=[chatbot], outputs=[msg, chatbot], queue=False).then(
               lambda: None, None, [msg], queue=False).then(
                   fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True)
    removelast.click(fn=remove_last, inputs=[chatbot], outputs=[chatbot], queue=False)
    clear.click(lambda: None, None, [chatbot], queue=False)

demo.queue(max_size=32, concurrency_count=2)
demo.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT, share=False)
