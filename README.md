```ascii
 █████╗ ██╗    ███████╗██████╗ ██╗  ██╗
██╔══██╗██║    ██╔════╝██╔══██╗██║ ██╔╝
███████║██║    ███████╗██║  ██║█████╔╝ 
██╔══██║██║    ╚════██║██║  ██║██╔═██╗ 
██║  ██║██║    ███████║██████╔╝██║  ██╗
╚═╝  ╚═╝╚═╝    ╚══════╝╚═════╝ ╚═╝  ╚═╝
```

# AI SDK

> [!WARNING]
> **This project is archived and no longer maintained.**
> Use Vercel's official [AI SDK for Python](https://ai-python.dev/docs) instead
> (`pip install ai`, source at [vercel-labs/ai-python](https://github.com/vercel-labs/ai-python)).
> The `ai-sdk-py` package on PyPI will not receive further updates.

A unified interface for working with different AI language model providers. Built with clean architecture and support for many providers

📚 [View the full documentation](https://jverre.github.io/ai-sdk/)

## Installation

```bash
pip install ai-sdk-py
```

## Quick Start

```python
from ai_sdk import generate_text
from ai_sdk.openai import openai

# Generate text
response = generate_text(
    model=openai("gpt-4o"),
    system="You are a helpful assistant.",
    prompt="What is the meaning of life?"
)

print(response.text)
```

## Features

- 🤖 Unified interface for multiple AI providers
- 🌟 Clean, modular architecture
- 🚀 Easy to extend with new providers
- 🛠️ Built-in support for function calling and tools
- 🖼️ Image generation and analysis capabilities
- 📋 Structured output generation with type validation

## Supported Providers

- Anthropic (Claude)
- OpenAI (GPT-4, GPT-3.5)
- More coming soon!

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

Created and maintained by @jverre.
