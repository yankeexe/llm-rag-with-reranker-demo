# 📑 RAG with Semantic Cache Demo Application

Demo RAG app with Semantic Caching for the YouTube video.

🚨 NOTE: **Requires `Python > 3.10` with  `SQLite > 3.35`**

Watch the video 👇

<a href="https://youtu.be/0agBo7_wKTo">
<img src="https://i.imgur.com/rJJaHmG.png" width="800">
</a>

## 🤖 Prerequisites

- [Ollama](https://ollama.dev/download)

## 🔨 Setting up locally

Create virtualenv and install dependencies.

```sh
make setup
```

## ⚡️ Running the application

```sh
make run
```

## ✨ Linters and Formatters

Check for linting rule violations:

```sh
make check
```

Auto-fix linting violations:

```sh
make fix
```

## 🤸‍♀️ Getting Help

```sh
make

# OR

make help
```

## 🔧 Common Issues and Fixes

- If you run into any errors with incompatible version of ChromaDB/Sqlite3, refer to [this solution](https://docs.trychroma.com/troubleshooting#sqlite).
