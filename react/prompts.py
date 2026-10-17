"""Few-shot prompts, modelled on the HotpotQA prompt from the ReAct paper.

The examples are stored once as data (question + steps) and turned into prompt
text by small functions. That keeps the ReAct, Act-only and CoT prompts in sync:
they all show the same questions, just in a different style.

The first three examples are shortened versions of the paper's own examples.
The fourth is new and shows the Calculate tool, which the paper doesn't have.
"""

REACT_INSTRUCTIONS = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be four types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Calculate[expression], which evaluates an arithmetic expression like 2023 - 1969.
(4) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""

# Each step is (thought, action, observation). The last step has observation None
# because Finish ends the episode.
EXAMPLES = [
    {
        "question": "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        "steps": [
            ("I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.",
             "Search[Colorado orogeny]",
             "The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas."),
            ("It does not mention the eastern sector. So I need to look up eastern sector.",
             "Lookup[eastern sector]",
             "(Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny."),
            ("The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.",
             "Search[High Plains]",
             "High Plains refers to one of two distinct land regions:"),
            ("I need to instead search High Plains (United States).",
             "Search[High Plains (United States)]",
             "The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m)."),
            ("High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.",
             "Finish[1,800 to 7,000 ft]",
             None),
        ],
        "cot": "The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.",
        "answer": "1,800 to 7,000 ft",
    },
    {
        "question": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?",
        "steps": [
            ("The question simplifies to \"The Simpsons\" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.",
             "Search[Milhouse]",
             "Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening."),
            ("The paragraph does not tell who Milhouse is named after, maybe I can look up \"named after\".",
             "Lookup[named after]",
             "(Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous."),
            ("Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.",
             "Finish[Richard Nixon]",
             None),
        ],
        "cot": "Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.",
        "answer": "Richard Nixon",
    },
    {
        "question": "Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?",
        "steps": [
            ("I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.",
             "Search[Adam Clayton Powell]",
             "Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito']."),
            ("To find the documentary, I can search Adam Clayton Powell (film).",
             "Search[Adam Clayton Powell (film)]",
             "Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr."),
            ("Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.",
             "Finish[The Saimaa Gesture]",
             None),
        ],
        "cot": "Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.",
        "answer": "The Saimaa Gesture",
    },
    {
        "question": "How many years after Harvard University was founded was Yale University founded?",
        "steps": [
            ("I need to find the founding years of Harvard University and Yale University, then subtract.",
             "Search[Harvard University]",
             "Harvard University is a private Ivy League research university in Cambridge, Massachusetts. Founded in 1636 as Harvard College, it is the oldest institution of higher learning in the United States."),
            ("Harvard was founded in 1636. Now I need to search Yale University.",
             "Search[Yale University]",
             "Yale University is a private Ivy League research university in New Haven, Connecticut. Founded in 1701, Yale is the third-oldest institution of higher education in the United States."),
            ("Yale was founded in 1701. I need to calculate 1701 - 1636.",
             "Calculate[1701 - 1636]",
             "65"),
            ("Yale was founded 65 years after Harvard, so the answer is 65.",
             "Finish[65]",
             None),
        ],
        "cot": "Harvard University was founded in 1636 and Yale University was founded in 1701. 1701 - 1636 = 65, so the answer is 65.",
        "answer": "65",
    },
]


def format_react_example(example: dict) -> str:
    """Write one example as numbered Thought / Action / Observation lines."""
    lines = [f"Question: {example['question']}"]
    for i, (thought, action, observation) in enumerate(example["steps"], start=1):
        lines.append(f"Thought {i}: {thought}")
        lines.append(f"Action {i}: {action}")
        if observation is not None:
            lines.append(f"Observation {i}: {observation}")
    return "\n".join(lines)


def build_react_prompt(question: str) -> str:
    """Instructions + examples + the new question, ready for the model to continue."""
    examples = "\n\n".join(format_react_example(ex) for ex in EXAMPLES)
    return f"{REACT_INSTRUCTIONS}\n{examples}\n\nQuestion: {question}\n"
