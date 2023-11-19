# Japanese LLM Simple WebUI
Rinna-3.6B、OpenCALM等の日本語対応LLM用の簡易Webチャットインタフェースです。
GGUF形式(llama.cpp)モデル用のものは、[GGUF Simple WebUI](https://github.com/noir55/gguf_simple_webui) になります。

# 機能
- ブラウザで開いたWebインタフェース上で、LLMとのチャットができます
- ストリーミング(生成中の表示)に対応
- LoRAの読み込みに対応
- [Rinna Japanese GPT NeoX 3.6B Instruction PPO](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)、[OpenCALM 7B](https://huggingface.co/cyberagent/open-calm-7b)、[Vicuna 7B](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1) で起動できることを確認しています
- [stablelm-tuned-alpha-chat](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat/tree/main) をベースに [Stability AIのチャットスクリプトを利用してRinnaのチャットモデルとお話する](https://nowokay.hatenablog.com/entry/2023/05/22/122040) などを参考に機能を追加しています

# 動作要件
- NVIDIA GPU、CUDA 環境前提です (GeForce RTX 3060、CUDA 11.7 環境で作成しています)
- Rocky Linux 8.7 上の Python 3.9.13 の環境で本スクリプトを作成していますが、おそらく [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) が動く環境であれば動作すると思います

# 使い方

## 起動方法

- `llm-webui.py` 内の20行目以降に各種設定項目がありますので、実行したいモデル、使用したいプロンプト、WebUIで使用するIPアドレスやポート番号などを記述して起動してください
```bash
$ python3 llm-webui.py
```
`llm-webui.py` のファイル名に指定はないため、ファイルを任意の名前でコピーして、モデルごとや設定ごとに使い分けることができます

- 実行時のオプションで `llm-webui.py` 内の設定を上書きできるため、コマンドオプションのみで設定を指定して起動することも可能です

Rinna 3.6B Instruction SFTモデルでの実行コマンド例
```bash
$ python3 llm-webui.py \
    --model rinna/japanese-gpt-neox-3.6b-instruction-sft-v2 \
    --model-type rinna \
    --tokenizer rinna/japanese-gpt-neox-3.6b-instruction-sft-v2 \
    --load-in-8bit off \
    --prompt-type rinna \
    --title "Rinna 3.6B Instruction SFT Chat"
```

OpenCALM 7Bモデルでの実行コマンド例
```bash
$ python3 llm-webui.py \
    --model cyberagent/open-calm-7b \
    --model-type opencalm \
    --tokenizer cyberagent/open-calm-7b \
    --load-in-8bit on \
    --prompt-type none \
    --title "Open CALM 7B Chat"
```

Rinna Bilingual GPT NeoX 4B Instruction SFTモデルでの実行コマンド例
```bash
$ python3 llm-webui.py \
    --model rinna/bilingual-gpt-neox-4b-instruction-sft \
    --model-type rinna4b \
    --tokenizer rinna/bilingual-gpt-neox-4b-instruction-sft \
    --load-in-8bit off \
    --prompt-type rinna \
    --title "Rinna Bilingual 4B Instruction SFT Chat"
```

Japanese StableLM Base Alpha 7Bモデルでの実行コマンド例
```bash
$ python3 llm-webui.py \
    --model stabilityai/japanese-stablelm-base-alpha-7b \
    --model-type ja-stablelm \
    --tokenizer novelai/nerdstash-tokenizer-v1 \
    --load-in-8bit on \
    --prompt-type none \
    --title "Japanese StableLM Chat"
```

- 起動したら、ブラウザで http://127.0.0.1:7860 (IPアドレス、ポート番号を変更した場合はそれに合わせてください)を開いてください

## 設定

以下のオプションが指定可能です。

| オプション                              | 説明                                                            |
| ------------------------------------ | ----------------------------------------------------------------- |
| --help                               | 指定可能なオプションの情報を出力する                                |
| --model <モデル名orパス>             | Huggingface上のモデル名または、保存したディレクトリのパスを指定する   |
| --model-type <モデルタイプ名>        | モデルタイプを指定する。タイプによりモデル読み込み方法、トークナイズ処理、改行の扱いなどが変わる |
| --tokenizer <トークナイザー名orパス> | トークナイザーのモデル名または、保存したディレクトリのパスを指定する。通常は `--model` で指定した値と同じでよい |
| --load-in-8bit <on/off>              | モデルを8ビット精度で実行する。GPUメモリ量の削減になる |
| --load-in-4bit <on/off>              | モデルを4ビット精度で実行する。GPUメモリ量の削減になる |
| --prompt-type <プロンプトタイプ名>   | 使用するプロンプトテンプレートを指定する。基本的にはモデルの学習に使用されたテンプレートを指定する。詳細は下の項目を参照 |
| --lora <LoRA保存ディレクトリパス>    | LoRAを読み込みたい場合、保存されたディレクトリのパスを指定する |
| --prompt-threshold <トークン数>      | プロンプト生成時に会話履歴を含めたトークン数がここで設定した数を超えると会話履歴が古い順に削除される |
| --prompt-deleted <トークン数>        | `--prompt-threshold` 設定値を超えて会話履歴が削除される場合、ここで指定したトークン数以下になるまで削除される |
| --max-new-tokens <トークン数>        | モデルが一度に生成する最大トークン数 |
| --temperature <Temperature値>        | 値を大きくすると多様な出力を行うようになる。0～1の間の値を設定する |
| --repetition-penalty <繰り返しペナルティ値>        | 値を大きくすると、繰り返しが発生しにくくなる。1.0だとペナルティなしとなる |
| --setting-visible <on/off>        | Advanced SettingsをWebUI上に表示するかどうか |
| --host <IPアドレス>                  | WebUIがバインドするアドレスを指定する。同じPC上のブラウザから使用する場合は `127.0.0.1` でよい |
| --port <ポート番号>                  | WebUIがバインドするポート番号を指定する。他のプログラムが使用していなければいくつでもよいが、Linux上で実行する場合、1024以下を指定するには通常root権限が必要 |
| --title "<タイトル文字列>"           | WebUIの最上部に表示するタイトルを任意に指定可能 |
| --debug <on/off>                     | コンソールにデバッグ情報(生成したプロンプト文字列や、トークナイズされた値など)を表示 |


## プロンプトタイプについて

モデルの学習時に使用されたプロンプトを指定することで精度の高い回答が期待できます

### プロンプトタイプ名 `none`
推奨モデル
 - Rinna Japanese GPT NeoX 3.6b
 - OpenCALM
 - StableLM Base Alpha
 - MPT 7B

プロンプトは使用せず、ユーザの入力した続きの文章をモデルが出力する形式。ファインチューニング前提の素のモデルを試す時などに使う
```
<ユーザの入力した文章><モデルの出力した文章>
```


### プロンプトタイプ名 `rinna`
推奨モデル
 - Rinna Japanese GPT NeoX 3.6b Instruction PPO
 - Rinna Japanese GPT NeoX 3.6b Instruction SFT V2
 - Rinna Bilingual GPT NeoX 4b Instruction PPO 
 - Rinna Bilingual GPT NeoX 4b Instruction SFT

プロンプト形式
```
ユーザー: <ユーザの入力した文章>
システム: <モデルの出力した文章>
```


### プロンプトタイプ名 `vicuna`
推奨モデル
 - Vicuna

プロンプト形式
```
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: <ユーザの入力した文章>
ASSISTANT: <モデルの出力した文章>
```


### プロンプトタイプ名 `alpaca`
推奨モデル
 - MPT 7B Instruct 
 - Alpaca-LoRAの学習スクリプトを使って学習されたモデル

プロンプト形式
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
<ユーザの入力した文章>

### Response:
<モデルの出力した文章>
```


### プロンプトタイプ名 `llama2`
推奨モデル
 - Llama 2 Chat

プロンプト形式
```
System: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
User: <ユーザの入力した文章>
Assistant: <モデルの出力した文章>
```


### プロンプトタイプ名 `beluga`
推奨モデル
 - Stable Beluga

プロンプト形式
```
### System:
You are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.

### User:
<ユーザの入力した文章>

### Assistant:
<モデルの出力した文章>
```


### プロンプトタイプ名 `ja-stablelm`
推奨モデル
 - Japanese StableLM Instruct Alpha

プロンプト形式
```
以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい

### 指示: <ユーザの入力した文章>
### 応答: <モデルの出力した文章>
```


### プロンプトタイプ名 `stablelm`
推奨モデル
 - StableLM Tuned Alpha

プロンプト形式
```
<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
<|USER|><ユーザの入力した文章><|ASSISTANT|><モデルの出力した文章>
```


### プロンプトタイプ名 `redpajama`
推奨モデル
 - MPT 7B Chat？(これが最適かどうか不明)

プロンプト形式
```
<human>: <ユーザの入力した文章>
<bot>: <モデルの出力した文章>
```


### プロンプトタイプ名 `xgen`
推奨モデル
 - XGen 7B 8K Inst

プロンプト形式
```
A chat between a curious human and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the human's questions.

### Human: <ユーザの入力した文章>
### Asisstant: <モデルの出力した文章>
```


### プロンプトタイプ名 `qa`
推奨モデル
 - Open LLaMA

プロンプト形式
```
Q: <ユーザの入力した文章>
A: <モデルの出力した文章>
```

# ライセンス

Japanese LLM Simple WebUI is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
