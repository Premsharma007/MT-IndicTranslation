module.exports = {
  run: [
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: { venv: "env" }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt --no-cache"
        ],
      }
    }
  ]
}
