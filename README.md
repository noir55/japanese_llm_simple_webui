# japanese_llm_simple_webui
Rinnna-3.6B、OpenCALM等の日本語対応LLM用の簡易Webインタフェースです

# 機能
- ブラウザで開いたWebインタフェースで、LLMとのチャットができます
- ストリーミング(生成中の表示)に対応しています
- LoRAにも対応しています
- Rinna-3.6B、OpenCALM-7B、Vicuna-7Bで起動できることを確認しています
- [stablelm-tuned-alpha-chat](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat/tree/main) をベースに [Stability AIのチャットスクリプトを利用してRinnaのチャットモデルとお話する](https://nowokay.hatenablog.com/entry/2023/05/22/122040) などを参考にさせていただき、機能を追加しています

# 動作要件
- NVIDIA GPU、CUDA 環境前提です
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) が動く環境であれば動作すると思います

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