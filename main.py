"""
Guessme.ai - Advanced AI-Powered Celebrity Guessing Game
======================================================

Built by Naveen Venkat (https://naveenvenkat.online/)

An advanced AI-powered celebrity guessing game featuring:
- Decoupled LLM architecture with specialized AI functions
- Strategic question generation and intelligent guessing
- Complete context awareness and past guess tracking
- 7 fixed questions for comprehensive celebrity profiling
- JSON format processing for structured data handling

Features:
- celebrity_guesser(): Makes strategic celebrity guesses
- question_generator(): Generates 3 strategic yes/no questions
- Past guesses tracking to avoid repetition
- Fixed responses grounding for better AI reasoning
- Comprehensive fallback systems

"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
import uuid
from typing import Dict
import logging
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Guessme.ai - Built by Naveen Venkat")

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# OpenAI configuration - Load from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Game state storage (in production, use Redis or database)
game_sessions: Dict[str, Dict] = {}

# Flow 2 specific questions (celebrity guessing game)

# Fixed questions for Flow 2 (first 7 questions) - all yes/no format
FIXED_QUESTIONS_FLOW2 = [
    "Is the person male?",
    "Is the person alive?",  # yes/no
    "Is the person real (not fictional)?",  # yes/no
    "Is the person an actor or actress?",  # yes/no
    "Is the person a sports person?",  # yes/no
    "Is the person American?",  # yes/no
    "Is the person Indian?"  # yes/no
]

# Pydantic models


class GameStart(BaseModel):
    flow: int = 2  # Only Flow 2 is supported


class QuestionAnswer(BaseModel):
    session_id: str
    answer: str  # "yes" or "no"


class GuessResponse(BaseModel):
    session_id: str
    is_correct: bool

# Routes


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/start_game")
async def start_game(game_data: GameStart):
    """Start a new celebrity guessing game"""
    session_id = str(uuid.uuid4())

    # Initialize Flow 2 game session
    game_sessions[session_id] = {
        "flow": 2,
        "game_over": False,
        "qa_history": [],
        "question_count": 0,
        "api_call_count": 0,
        "phase": "confirmation",  # confirmation -> fixed_questions -> ai_guessing
        "fixed_questions_completed": 0,
        "past_guesses": [],  # Store wrong guesses to avoid repeating them
        "current_ai_guess": None  # Store the current AI guess for tracking
    }

    return {
        "session_id": session_id,
        "message": "Think of any celebrity in your mind!",
        "question": "Have you thought of a celebrity?",
        "phase": "confirmation"
    }


@app.post("/answer_question")
async def answer_question(answer_data: QuestionAnswer):
    session_id = answer_data.session_id
    answer = answer_data.answer.lower().strip()

    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = game_sessions[session_id]

    if session["flow"] != 2:
        raise HTTPException(
            status_code=400, detail="Invalid flow for this endpoint")

    if session["game_over"]:
        return {"game_over": True, "message": "Game already completed"}

    # Handle confirmation phase
    if session["phase"] == "confirmation":
        if answer == "yes":
            session["phase"] = "fixed_questions"
            session["question_count"] = 1
            session["qa_history"].append({
                "question": "Have you thought of a celebrity?",
                "answer": "yes"
            })
            return {
                "type": "question",
                "question": FIXED_QUESTIONS_FLOW2[0],
                "question_number": 1,
                "phase": "fixed_questions"
            }
        else:
            return {
                "type": "question",
                "question": "Please think of a celebrity first, then click Yes when ready.",
                "phase": "confirmation"
            }

    # Handle fixed questions phase (first 5 questions)
    elif session["phase"] == "fixed_questions":
        current_question = FIXED_QUESTIONS_FLOW2[session["fixed_questions_completed"]]
        session["qa_history"].append({
            "question": current_question,
            "answer": answer
        })
        session["fixed_questions_completed"] += 1
        session["question_count"] += 1

        # Check if we've completed all 7 fixed questions
        if session["fixed_questions_completed"] >= 7:
            session["phase"] = "ai_guessing"
            # Make first AI guess based on the 7 fixed questions
            try:
                session["api_call_count"] += 1
                # Extract the 7 fixed question responses as variables
                fixed_responses = extract_fixed_responses(
                    session["qa_history"])

                # First call: Get celebrity guess only
                celebrity_guess = await celebrity_guesser(session["qa_history"], session["past_guesses"], fixed_responses)

                # Store the current AI guess for tracking
                session["current_ai_guess"] = celebrity_guess["guess"]

                return {
                    "type": "guess",
                    "guess": celebrity_guess["guess"],
                    "api_calls_used": session["api_call_count"]
                }
            except Exception as e:
                return {"error": "Failed to get AI response", "details": str(e)}
        else:
            # Continue with next fixed question
            next_question = FIXED_QUESTIONS_FLOW2[session["fixed_questions_completed"]]
            return {
                "type": "question",
                "question": next_question,
                "question_number": session["question_count"] + 1,
                "phase": "fixed_questions"
            }

    # Handle AI guessing phase (after fixed questions)
    elif session["phase"] == "ai_guessing":
        # Store the answer to current question
        if "current_question" in session:
            session["qa_history"].append({
                "question": session["current_question"],
                "answer": answer
            })
            session["question_count"] += 1

        # Check if we've reached the API call limit
        if session["api_call_count"] >= 10:
            session["game_over"] = True
            return {
                "game_over": True,
                "message": "I give up! You've stumped me after 10 attempts. What was the answer?"
            }

        # Get next AI guess and questions
        try:
            session["api_call_count"] += 1
            # Extract the 7 fixed question responses as variables
            fixed_responses = extract_fixed_responses(session["qa_history"])

            # Get celebrity guess only
            celebrity_guess = await celebrity_guesser(session["qa_history"], session["past_guesses"], fixed_responses)

            # Store the current AI guess for tracking
            session["current_ai_guess"] = celebrity_guess["guess"]

            return {
                "type": "guess",
                "guess": celebrity_guess["guess"],
                "api_calls_used": session["api_call_count"]
            }
        except Exception as e:
            return {"error": "Failed to get AI response", "details": str(e)}

    return {"error": "Invalid game state"}


@app.post("/respond_to_ai_guess")
async def respond_to_ai_guess(response_data: GuessResponse):
    session_id = response_data.session_id
    is_correct = response_data.is_correct

    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = game_sessions[session_id]

    if session["flow"] != 2:
        raise HTTPException(
            status_code=400, detail="Invalid flow for this endpoint")

    if is_correct:
        session["game_over"] = True
        # Clear the current guess since it was correct
        session["current_ai_guess"] = None
        return {"game_over": True, "message": "🎉 I guessed it correctly! Thanks for playing!"}
    else:
        # Add the wrong guess to past_guesses to avoid repeating it
        if session["current_ai_guess"] and session["current_ai_guess"] not in session["past_guesses"]:
            session["past_guesses"].append(session["current_ai_guess"])
            print(
                f"📝 Added '{session['current_ai_guess']}' to past guesses. Total past guesses: {len(session['past_guesses'])}")

        # Generate 3 new questions based on current context
        try:
            session["api_call_count"] += 1
            # Extract the 7 fixed question responses as variables
            fixed_responses = extract_fixed_responses(session["qa_history"])
            questions_result = await question_generator(session["qa_history"], session["past_guesses"], fixed_responses)

            session["current_questions"] = questions_result["questions"]
            session["question_index"] = 0

            return {
                "type": "question",
                "question": questions_result["questions"][0],
                "question_number": session["question_count"] + 1,
                "remaining_questions": len(questions_result["questions"])
            }
        except Exception as e:
            return {"error": "Failed to generate questions", "details": str(e)}
        else:
            session["game_over"] = True
            return {"game_over": True, "message": "I give up! You stumped me! What was the answer?"}


@app.post("/continue_questions")
async def continue_questions(answer_data: QuestionAnswer):
    """Handle answers to the 3 follow-up questions after a wrong guess"""
    session_id = answer_data.session_id
    answer = answer_data.answer.lower().strip()

    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = game_sessions[session_id]

    if session["flow"] != 2:
        raise HTTPException(
            status_code=400, detail="Invalid flow for this endpoint")

    if session["game_over"]:
        return {"game_over": True, "message": "Game already completed"}

    # Store the answer
    current_question = session["current_questions"][session["question_index"]]
    session["qa_history"].append({
        "question": current_question,
        "answer": answer
    })
    session["question_count"] += 1
    session["question_index"] += 1

    # Check if we still have questions in current batch
    if session["question_index"] < len(session["current_questions"]):
        next_question = session["current_questions"][session["question_index"]]
        return {
            "type": "question",
            "question": next_question,
            "question_number": session["question_count"] + 1,
            "remaining_questions": len(session["current_questions"]) - session["question_index"]
        }
    else:
        # Completed all 3 questions, now make another guess
        # But don't make a guess immediately, store the current question for processing
        session["current_question"] = current_question
        return {
            "type": "ready_for_guess",
            "message": "Ready to make next guess based on your answers"
        }


def extract_fixed_responses(qa_history: list[dict]) -> dict:
    """Extract the 7 fixed question responses into structured variables"""
    # Create a mapping from questions to their answers
    qa_dict = {qa['question']: qa['answer'].lower() for qa in qa_history}

    # Extract the 7 fixed question responses
    gender = "male" if qa_dict.get(
        "Is the person male?", "").lower() == "yes" else "female"
    alive = "Yes" if qa_dict.get(
        "Is the person alive?", "").lower() == "yes" else "No"
    real_or_fictional = "Real" if qa_dict.get(
        "Is the person real (not fictional)?", "").lower() == "yes" else "Fictional"
    actor_actress = "Yes" if qa_dict.get(
        "Is the person an actor or actress?", "").lower() == "yes" else "No"
    sports_person = "Yes" if qa_dict.get(
        "Is the person a sports person?", "").lower() == "yes" else "No"
    american = "Yes" if qa_dict.get(
        "Is the person American?", "").lower() == "yes" else "No"
    indian = "Yes" if qa_dict.get(
        "Is the person Indian?", "").lower() == "yes" else "No"

    fixed_responses = {
        "gender": gender,
        "alive": alive,
        "real_or_fictional": real_or_fictional,
        "actor_actress": actor_actress,
        "sports_person": sports_person,
        "american": american,
        "indian": indian
    }

    print(f"🔍 Extracted Fixed Question Responses:")
    for key, value in fixed_responses.items():
        print(f"   {key}: {value}")
    print()

    return fixed_responses


async def celebrity_guesser(qa_history: list[dict], past_guesses: list[str] = None, fixed_responses: dict = None) -> dict:
    """Generate only a celebrity guess based on Q&A history"""
    print("\n🎯 Starting Celebrity Guesser...")

    try:
        # Create OpenAI client
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Convert Q&A history to JSON format
        history_json = {qa['question']: qa['answer'] for qa in qa_history}
        history_text = json.dumps(history_json, indent=2)

        print(f"📋 Q&A History in JSON format ({len(history_json)} questions):")
        for question, answer in history_json.items():
            print(f"   \"{question}\": \"{answer}\"")
        print()

        # Handle past guesses
        if past_guesses is None:
            past_guesses = []
        if past_guesses:
            print(f"❌ Past wrong guesses ({len(past_guesses)}):")
            for i, guess in enumerate(past_guesses, 1):
                print(f"   {i}. {guess}")
            print()

        prompt = f"""
You are a celebrity guessing expert. Based on the Q&A history provided, make your BEST guess for which celebrity the user is thinking of.

IMPORTANT RULES:
1. Provide ONLY ONE celebrity guess (full name)
2. Consider all the Q&A history carefully
3. DO NOT guess any celebrity from the "Previously guessed celebrities" list
4. Use the fixed question responses to guide your guess

Input:
Complete game history in JSON format:
{history_text}

Fixed Question Responses (use these to ground your guess):
Gender: {fixed_responses.get('gender', 'Unknown') if fixed_responses else 'Unknown'}
Alive: {fixed_responses.get('alive', 'Unknown') if fixed_responses else 'Unknown'}
Real or Fictional: {fixed_responses.get('real_or_fictional', 'Unknown') if fixed_responses else 'Unknown'}
Actor/Actress: {fixed_responses.get('actor_actress', 'Unknown') if fixed_responses else 'Unknown'}
Sports Person: {fixed_responses.get('sports_person', 'Unknown') if fixed_responses else 'Unknown'}
American: {fixed_responses.get('american', 'Unknown') if fixed_responses else 'Unknown'}
Indian: {fixed_responses.get('indian', 'Unknown') if fixed_responses else 'Unknown'}

Previously guessed celebrities (DO NOT guess these again):
{', '.join(past_guesses) if past_guesses else 'None - this is your first guess'}

Respond with ONLY the celebrity's full name, nothing else.
"""

        print("\n🤖 CELEBRITY GUESSER PROMPT (BEFORE API CALL):")
        print("="*80)
        print(prompt)
        print("="*80)
        print()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,
        )

        guess = response.choices[0].message.content.strip()
        print("🎯 CELEBRITY GUESSER RESPONSE:")
        print("=" * 80)
        print(f"Guess: {guess}")
        print("=" * 80)

        return {"guess": guess}

    except Exception as e:
        logger.error(f"Error in celebrity_guesser: {e}")
        # Fallback guess based on fixed responses
        fallback_guess = "Taylor Swift"  # Default fallback
        if fixed_responses:
            if fixed_responses.get('gender') == 'male':
                if fixed_responses.get('sports_person') == 'Yes':
                    fallback_guess = "LeBron James"
                elif fixed_responses.get('actor_actress') == 'Yes':
                    fallback_guess = "Leonardo DiCaprio"
                else:
                    fallback_guess = "Tom Hanks"
            else:  # female
                if fixed_responses.get('sports_person') == 'Yes':
                    fallback_guess = "Serena Williams"
                elif fixed_responses.get('actor_actress') == 'Yes':
                    fallback_guess = "Emma Stone"
                else:
                    fallback_guess = "Taylor Swift"

        return {"guess": fallback_guess}


async def question_generator(qa_history: list[dict], past_guesses: list[str] = None, fixed_responses: dict = None) -> dict:
    """Generate 3 strategic yes/no questions based on Q&A history"""
    print("\n❓ Starting Question Generator...")

    try:
        # Create OpenAI client
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Convert Q&A history to JSON format
        history_json = {qa['question']: qa['answer'] for qa in qa_history}
        history_text = json.dumps(history_json, indent=2)

        # Handle past guesses
        if past_guesses is None:
            past_guesses = []
        if past_guesses:
            print(f"❌ Past wrong guesses ({len(past_guesses)}):")
            for i, guess in enumerate(past_guesses, 1):
                print(f"   {i}. {guess}")
            print()

        prompt = f"""
You are an expert at generating strategic yes/no questions for celebrity guessing games. Based on the current Q&A history, generate EXACTLY 3 new yes/no questions that will help identify the mystery celebrity.

IMPORTANT RULES:
1. Generate EXACTLY 3 questions
2. All questions must be yes/no format
3. DO NOT repeat any questions from the game history
4. Questions should be strategic and help narrow down possibilities
5. Consider the fixed question responses when creating questions
6. Questions should be mutually exclusive and help eliminate options

Input:
Complete game history in JSON format:
{history_text}

Fixed Question Responses:
Gender: {fixed_responses.get('gender', 'Unknown') if fixed_responses else 'Unknown'}
Alive: {fixed_responses.get('alive', 'Unknown') if fixed_responses else 'Unknown'}
Real or Fictional: {fixed_responses.get('real_or_fictional', 'Unknown') if fixed_responses else 'Unknown'}
Actor/Actress: {fixed_responses.get('actor_actress', 'Unknown') if fixed_responses else 'Unknown'}
Sports Person: {fixed_responses.get('sports_person', 'Unknown') if fixed_responses else 'Unknown'}
American: {fixed_responses.get('american', 'Unknown') if fixed_responses else 'Unknown'}
Indian: {fixed_responses.get('indian', 'Unknown') if fixed_responses else 'Unknown'}

Previously guessed celebrities (DO NOT generate questions about these):
{', '.join(past_guesses) if past_guesses else 'None - no wrong guesses yet'}

Previously asked questions (DO NOT repeat):
{chr(10).join([f"- {qa['question']}" for qa in qa_history])}

Respond in JSON format:
{{
    "questions": [
        "Your first strategic yes/no question?",
        "Your second strategic yes/no question?",
        "Your third strategic yes/no question?"
    ]
}}
"""

        print("\n🤖 QUESTION GENERATOR PROMPT (BEFORE API CALL):")
        print("="*80)
        print(prompt)
        print("="*80)
        print()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )

        content = response.choices[0].message.content
        print("❓ QUESTION GENERATOR RAW RESPONSE:")
        print("=" * 80)
        print(repr(content))
        print("=" * 80)
        print("❓ QUESTION GENERATOR FORMATTED RESPONSE:")
        print("=" * 80)
        print(content)
        print("=" * 80)

        # Parse JSON response
        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                try:
                    result = json.loads(json_content)
                    print("✅ Successfully parsed JSON from response")
                except json.JSONDecodeError:
                    print("❌ Still failed to parse JSON")
                    raise ValueError(
                        f"Could not parse JSON from AI response: {content[:200]}...")
            else:
                raise ValueError(
                    f"No JSON found in AI response: {content[:200]}...")

        if "questions" not in result or not isinstance(result["questions"], list) or len(result["questions"]) != 3:
            raise ValueError(
                "Question generator must return exactly 3 questions")

        return result

    except Exception as e:
        logger.error(f"Error in question_generator: {e}")
        # Fallback questions based on fixed responses
        fallback_questions = [
            "Is this person from the United States?",
            "Has this person won any major awards?",
            "Is this person over 40 years old?"
        ]

        if fixed_responses:
            if fixed_responses.get('actor_actress') == 'Yes':
                fallback_questions = [
                    "Has this person been in Marvel movies?",
                    "Has this person won an Oscar?",
                    "Is this person known for comedy roles?"
                ]
            elif fixed_responses.get('sports_person') == 'Yes':
                fallback_questions = [
                    "Is this person known for basketball?",
                    "Has this person won an Olympic gold medal?",
                    "Is this person retired?"
                ]

        return {"questions": fallback_questions}

    try:
        # Create OpenAI client without any proxy configurations
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Convert Q&A history to JSON format
        history_json = {qa['question']: qa['answer'] for qa in qa_history}
        history_text = json.dumps(history_json, indent=2)

        print(f"📋 Q&A History in JSON format ({len(history_json)} questions):")
        for question, answer in history_json.items():
            print(f"   \"{question}\": \"{answer}\"")
        print()

        # Handle past guesses
        if past_guesses is None:
            past_guesses = []
        if past_guesses:
            print(f"❌ Past wrong guesses ({len(past_guesses)}):")
            for i, guess in enumerate(past_guesses, 1):
                print(f"   {i}. {guess}")
            print()
        else:
            print("✅ No past wrong guesses yet")
            print()

        prompt = f"""
You are the master at a celebrity guessing game. Your role is to ask **Yes/No questions** and make a **specific guess** about the celebrity. Use the complete Q&A history provided to you.

Rules:
1. Always provide **one best guess** (a real celebrity’s full name).  
2. Provide **exactly 3 new Yes/No questions** that are strategic, mutually exclusive, and collectively exhaustive.  
   - Do NOT repeat any previously asked questions.  
   - Assume your current guess is wrong, so frame the 3 questions to identify the next celebrity.  
   - Questions must not depend on the guessed celebrity.  
3. If the user has already said “Not Male,” you must treat the celebrity as Female.  
4. Your output must be **valid JSON only**, with no extra text, comments, or explanations.  

Input:
Complete game history in JSON format:
{history_text}

Fixed Question Responses (use these to ground your guess):
Gender: {fixed_responses.get('gender', 'Unknown') if fixed_responses else 'Unknown'}
Alive: {fixed_responses.get('alive', 'Unknown') if fixed_responses else 'Unknown'}
Real or Fictional: {fixed_responses.get('real_or_fictional', 'Unknown') if fixed_responses else 'Unknown'}
Actor/Actress: {fixed_responses.get('actor_actress', 'Unknown') if fixed_responses else 'Unknown'}
Sports Person: {fixed_responses.get('sports_person', 'Unknown') if fixed_responses else 'Unknown'}
American: {fixed_responses.get('american', 'Unknown') if fixed_responses else 'Unknown'}
Indian: {fixed_responses.get('indian', 'Unknown') if fixed_responses else 'Unknown'}

Previously guessed celebrities (DO NOT guess these again):
{', '.join(past_guesses) if past_guesses else 'None - this is your first guess'}

Output format (strict JSON only):
{{
    "guess": "Celebrity Full Name",
    "questions": [
        "Is this person known for [specific trait/work]?",
        "Has this person [specific achievement/characteristic]?",
        "Does this person [specific detail]?"
    ]
}}
"""

        print("\n" + "="*80)
        print("🤖 LLM PROMPT (BEFORE API CALL):")
        print("="*80)
        print(prompt)
        print("="*80)
        print()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        content = response.choices[0].message.content
        print("🔥 LLM RAW RESPONSE:")
        print("=" * 80)
        print(repr(content))  # Show raw content with quotes for debugging
        print("=" * 80)
        print("🔥 LLM FORMATTED RESPONSE:")
        print("=" * 80)
        print(content)
        print("=" * 80)

        # Try to parse JSON with better error handling
        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print("🔧 Attempting to extract JSON from response...")

            # Try to find JSON in the response (between { and })
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                print(f"📋 Found potential JSON: {json_content}")
                try:
                    result = json.loads(json_content)
                    print("✅ Successfully parsed JSON from response")
                except json.JSONDecodeError as e2:
                    print(f"❌ Still failed to parse JSON: {e2}")
                    raise ValueError(
                        f"Could not parse JSON from AI response: {content[:200]}...")
            else:
                print("❌ No JSON found in response")
                raise ValueError(
                    f"No JSON found in AI response: {content[:200]}...")

        if "guess" not in result or "questions" not in result:
            raise ValueError("Invalid response format")
        if not isinstance(result["questions"], list) or len(result["questions"]) != 3:
            raise ValueError("Must have exactly 3 questions")

        return result

    except Exception as e:
        logger.error(f"Error in question_generator: {e}")


@app.delete("/quit_game/{session_id}")
async def quit_game(session_id: str):
    if session_id in game_sessions:
        del game_sessions[session_id]
    return {"message": "Game quit successfully"}


@app.get("/flow2", response_class=HTMLResponse)
async def flow2(request: Request):
    return templates.TemplateResponse("flow2.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
