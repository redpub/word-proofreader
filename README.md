# 📝 AI Word Proofreader

A Streamlit application that uses AI (via OpenRouter) to proofread Word documents and apply corrections as tracked changes, powered by the [docx-revisions](https://balalofernandez.github.io/docx-revisions/) library.

## Features

- 🤖 **AI-Powered Proofreading**: Uses LLMs to detect and correct grammar, spelling, style, and clarity issues
- 📝 **Tracked Changes**: All corrections appear as tracked changes in Microsoft Word
- 🔧 **Customizable**: Choose from multiple AI models and customize the proofreading prompt
- 🎯 **User-Friendly**: Simple drag-and-drop interface with real-time feedback
- 📊 **Detailed Reports**: See exactly what was changed and why

## Installation

1. Clone this repository or download the files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Get an API key from [OpenRouter](https://openrouter.ai/keys)

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

3. In the app:
   - Enter your OpenRouter API key in the sidebar
   - Select your preferred AI model
   - (Optional) Customize the system prompt
   - Upload a Word document (.docx)
   - Click "Start Proofreading"
   - Review the suggested edits
   - Download the proofread document
   - Open in Microsoft Word to review/accept/reject tracked changes

## Supported Models

The app supports various models through OpenRouter:

- **OpenAI**: GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku
- **Google**: Gemini Pro 1.5
- **Meta**: Llama 3.1 (70B, 8B)
- **Mistral**: Mistral Large

## How It Works

1. **Document Upload**: The app loads your Word document using `docx-revisions`
2. **AI Analysis**: The selected LLM analyzes each paragraph for improvements
3. **Structured Edits**: The AI returns corrections in JSON format with reasons
4. **Tracked Changes**: Edits are applied using `replace_tracked()` and `add_tracked_insertion()`
5. **Download**: The modified document is saved with all changes tracked
6. **Review in Word**: Open the document in Microsoft Word to see red-line changes

## Configuration

### System Prompt

Customize the proofreading instructions by editing the system prompt in the sidebar. The default prompt focuses on:
- Spelling errors
- Grammar mistakes
- Punctuation issues
- Style improvements
- Clarity and readability

### Author Name

Set the author name that will appear in the tracked changes (default: "AI Proofreader")

## Example Workflow

```python
# The app follows this pattern:
1. Load document with RevisionDocument
2. Extract paragraphs with indices
3. Send to LLM for analysis
4. Receive structured edits (paragraph_index, search_text, new_text, reason)
5. Apply edits using replace_tracked() or add_tracked_insertion()
6. Save document with tracked changes
```

## Technical Details

- **Framework**: Streamlit
- **AI Integration**: OpenRouter API (OpenAI-compatible)
- **Document Processing**: docx-revisions library
- **Data Validation**: Pydantic models

## Credits

Built with:
- [docx-revisions](https://github.com/balalofernandez/docx-revisions) - Python library for Word tracked changes
- [Streamlit](https://streamlit.io/) - Web app framework
- [OpenRouter](https://openrouter.ai/) - Unified API for LLMs

## License

See LICENSE file for details.
