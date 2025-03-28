```ascii
 █████╗ ██╗    ███████╗██████╗ ██╗  ██╗
██╔══██╗██║    ██╔════╝██╔══██╗██║ ██╔╝
███████║██║    ███████╗██║  ██║█████╔╝ 
██╔══██║██║    ╚════██║██║  ██║██╔═██╗ 
██║  ██║██║    ███████║██████╔╝██║  ██╗
╚═╝  ╚═╝╚═╝    ╚══════╝╚═════╝ ╚═╝  ╚═╝
```

# AI SDK

A unified interface for working with different AI language model providers. Built with clean architecture and support for many providers

## Installation

```bash
pip install ai-sdk
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







[project]
name = "ai-sdk-py"
version = "0.1.0"
description = ""
authors = [
    {name = "Jacques", email = "jverre@gmail.com"}
]
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "pydantic>=2.10.6,<3.0.0",
    "httpx>=0.28.1,<0.29.0",
    "validators>=0.34.0,<0.35.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "python-dotenv>=1.0.0"
]

[project.urls]
Homepage = "https://github.com/jverre/ai-sdk"
Repository = "https://github.com/jverre/ai-sdk"

[project.classifiers]
Development Status :: 3 - Alpha
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.9
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Scientific/Engineering :: Artificial Intelligence

[tool.setuptools]
packages = [{include = "ai_sdk", from = "src"}]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v -s"
