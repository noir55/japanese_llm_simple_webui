# japanese_llm_simple_webui
Rinna-3.6B、OpenCALM等の日本語対応LLM用の簡易Webチャットインタフェースです

# 機能
- ブラウザで開いたWebインタフェース上で、LLMとのチャットができます
- ストリーミング(生成中の表示)に対応
- LoRAの読み込みに対応
- [Rinna Japanese GPT NeoX 3.6B Instruction PPO](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)、[OpenCALM 7B](https://huggingface.co/cyberagent/open-calm-7b)、[Vicuna 7B](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1) で起動できることを確認しています
- [stablelm-tuned-alpha-chat](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat/tree/main) をベースに [Stability AIのチャットスクリプトを利用してRinnaのチャットモデルとお話する](https://nowokay.hatenablog.com/entry/2023/05/22/122040) などを参考にさせていただき、機能を追加しています

# 動作要件
- NVIDIA GPU、CUDA 環境前提です
- Rocky Linux 8.7 上の Python 3.9.13 の環境で本スクリプトを作成していますが、おそらく [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) が動く環境であれば動作すると思います

# 使い方
- `llm-webui.py` 内の20～54行目あたりに各種設定項目がありますので、実行したいモデル、使用したいプロンプト、WebUIで使用するIPアドレスやポート番号などを記述して起動してください
```bash
$ python3 llm-webui.py
```
- 実行時のオプションで `llm-webui.py` 内の設定を上書きできます。指定可能なオプションは以下のように `--help` オプションを付けてコマンドを実行して確認してください
```bash
$ python3 llm-webui.py --help
```
- `llm-webui.py` のファイル名に指定はないため、ファイルを任意の名前でコピーして、モデルごとや設定ごとに使い分けることができます