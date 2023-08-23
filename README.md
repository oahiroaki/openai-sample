# 環境構築(Mac)

## Xcodeコマンドラインツールのインストール

次にインストールするHomebrewのために必要なのでインストールする。  
これをインストールすることでgitなども同時にインストールされる。

Terminalを起動して、以下のコマンドを実行する。

```
$ xcode-select --install
```

## Homebrewのインストール

HomebrewはMacOS用のパッケージマネージャ。  
[公式サイト](https://docs.brew.sh/Installation)に記載の通りにインストールする。

Terminalを起動して、以下のコマンドを実行する。

```bash
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Pythonのインストール

Homebrewを使って以下のようにインストールする。

```
$ brew install python
```

インストール後、以下のようにバージョン表示されることを確認する。

```
$ python3 --version
> Python 3.11.4
```

## streamlitのインストール

```
pip3 install -r requirements.txt
```

