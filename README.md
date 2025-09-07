# Guessme.ai 🧞‍♂️

An advanced AI-powered celebrity guessing game featuring a sophisticated decoupled LLM architecture. Think of any celebrity and watch as our AI uses strategic questioning and intelligent reasoning to guess who you're thinking of!

## 🎯 Game Features

### 🤖 Advanced AI Architecture
- **Decoupled LLM System**: Two specialized AI functions for optimal performance
  - `celebrity_guesser()` - Makes strategic celebrity guesses
  - `question_generator()` - Generates 3 strategic yes/no questions per round

### 🎮 Game Flow
- Think of any celebrity in your mind
- Answer **7 strategic fixed questions** to establish baseline characteristics
- AI makes educated guesses based on your answers
- If wrong, AI generates 3 new strategic questions for the next round
- **10 total attempts** for the AI to guess correctly
- **Past guesses tracking** prevents AI from repeating wrong answers

### 📋 Fixed Question Set (7 Questions)
1. **Gender**: Is the person male?
2. **Vital Status**: Is the person alive?
3. **Reality**: Is the person real (not fictional)?
4. **Profession**: Is the person an actor or actress?
5. **Sports**: Is the person a sports person?
6. **Nationality**: Is the person American?
7. **Nationality**: Is the person Indian?

### 🧠 AI Intelligence Features
- **Complete Context Awareness**: All Q&A history from all rounds included
- **JSON Format Processing**: Structured data for better AI reasoning
- **Grounding Values**: Fixed question responses guide AI reasoning
- **Past Guesses Prevention**: AI won't repeat previously guessed celebrities
- **Strategic Questioning**: AI generates mutually exclusive, exhaustive questions
- **Fallback Systems**: Robust error handling with intelligent defaults

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd guessme-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8000`

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML + Tailwind CSS + JavaScript
- **AI**: OpenAI GPT-4o-mini (optimized for strategic reasoning)
- **State Management**: In-memory sessions with UUID4 security
- **Data Format**: JSON for structured Q&A processing
- **Error Handling**: Comprehensive fallback systems

## 📁 Project Structure

```
guessme-ai/
├── main.py              # FastAPI application with decoupled AI
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create from .env.example)
├── env_example.txt      # Environment variables template
├── README.md           # This file
├── render.yaml         # Render deployment config
├── templates/          # HTML templates
│   ├── base.html      # Base template
│   ├── index.html     # Landing page
│   └── flow2.html     # Celebrity guessing game
└── static/            # Static files (CSS/JS)
```

## 🌐 API Endpoints

### Core Game Endpoints
- `GET /` - Landing page
- `GET /flow2` - Celebrity guessing game page
- `POST /start_game` - Initialize new game session
- `POST /answer_question` - Answer AI's yes/no questions
- `POST /respond_to_ai_guess` - Respond to AI's celebrity guess (yes/no)
- `POST /continue_questions` - Answer follow-up questions after wrong guess
- `DELETE /quit_game/{session_id}` - End game session

### AI Functions (Internal)
- `celebrity_guesser()` - Makes strategic celebrity guesses
- `question_generator()` - Generates 3 strategic yes/no questions
- `extract_fixed_responses()` - Processes Q&A into structured variables

## 🎯 AI Architecture Deep Dive

### 🔄 Game Flow Architecture
```
1. User thinks of celebrity
2. User answers 7 fixed questions
3. celebrity_guesser() makes first guess
4. If wrong: question_generator() creates 3 questions
5. User answers 3 questions
6. celebrity_guesser() makes second guess
7. Repeat until correct or 10 attempts reached
```

### 🧠 AI Intelligence Features

#### **Complete Context Awareness**
- **All Q&A History**: Every question and answer from all rounds
- **Past Guesses**: List of previously guessed celebrities (avoided)
- **Fixed Responses**: Structured data for grounding AI reasoning

#### **Strategic Question Generation**
- **Mutually Exclusive**: Questions don't overlap in scope
- **Exhaustive Coverage**: Covers different aspects systematically
- **Context-Aware**: Considers fixed responses for relevant questions

#### **Intelligent Guess Prevention**
- **Past Guesses Tracking**: AI won't repeat wrong guesses
- **Nationality Awareness**: Considers American/Indian status
- **Professional Context**: Uses actor/sports person status

### 📊 Data Flow
```
User Input → Fixed Questions → JSON Processing → AI Context → Strategic Guessing
                                      ↓
                            Question Generation → Follow-up Questions
```

## 🚀 Deployment on Render

### Method 1: Using render.yaml (Recommended)

1. Fork this repository to your GitHub account
2. Connect your GitHub account to Render
3. Create a new Web Service from your repository
4. Render will automatically detect the `render.yaml` file
5. Set the `OPENAI_API_KEY` environment variable in Render dashboard
6. Deploy!

### Method 2: Manual Setup

1. Create a new Web Service on Render
2. Connect your repository
3. Configure the service:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**:
     - `OPENAI_API_KEY`: Your OpenAI API key

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Game Configuration
Edit `main.py` to modify:

- **Fixed Questions**: Modify `FIXED_QUESTIONS_FLOW2` list
- **AI Model**: Change model in `celebrity_guesser()` and `question_generator()`
- **Max Attempts**: Change attempt limits in game logic
- **Fallback Behavior**: Modify fallback guess/question logic

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Modern UI**: Tailwind CSS with gradients and animations
- **Real-time Feedback**: Live updates during gameplay
- **Loading States**: Smooth transitions and indicators
- **Game Statistics**: API call tracking and attempt counting

## 🔒 Security & Performance

- **Environment Variables**: API keys never exposed to frontend
- **Session Security**: UUID4-generated session IDs
- **Error Handling**: Comprehensive fallback systems
- **Optimized AI Calls**: Decoupled architecture reduces token usage
- **JSON Processing**: Efficient structured data handling

## 🧪 Testing

Run the comprehensive test suite:
```bash
python -c "from main import app; print('✅ App loads successfully')"
```

The application includes built-in logging to show:
- Q&A history processing
- AI prompt generation
- API call responses
- Guess tracking and prevention

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (the app includes comprehensive logging)
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:

1. Check that your OpenAI API key is correctly set in `.env`
2. Ensure all dependencies are installed
3. Check the console logs for detailed AI processing information
4. Verify your celebrity choice is well-known (the AI works best with famous celebrities)

## 🎯 Advanced Features

### **Decoupled AI Architecture Benefits**
- **Specialized Roles**: Each AI function has a focused responsibility
- **Better Performance**: Optimized prompts for specific tasks
- **Improved Accuracy**: Context-aware reasoning with complete history
- **Strategic Depth**: Intelligent question generation based on fixed responses
- **Memory Efficiency**: Past guesses tracking prevents repetition

### **Grounding Values System**
The AI receives structured grounding data:
```json
{
  "gender": "male",
  "alive": "Yes",
  "real_or_fictional": "Real",
  "actor_actress": "No",
  "sports_person": "Yes",
  "american": "No",
  "indian": "Yes"
}
```

This ensures the AI makes informed, contextually appropriate guesses and questions.

---

**Built with ❤️ by [Naveen Venkat](https://naveenvenkat.online/)** 🧠✨

Enjoy playing the most advanced AI celebrity guessing game! 🎭
