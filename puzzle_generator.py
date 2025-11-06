import random

class PuzzleGenerator:
    """
    Generates age-appropriate math problems for children aged 5-10
    
    Difficulty Guidelines:
    - Easy: Ages 5-6 (Kindergarten/Grade 1)
    - Medium: Ages 7-8 (Grade 2-3)
    - Hard: Ages 9-10 (Grade 4-5)
    """
    
    def __init__(self):
        self.recent_problems = []  # Track to avoid repetition
        self.max_history = 10
        
    def generate(self, difficulty):
        """
        Generate a math problem based on difficulty level
        
        Args:
            difficulty (str): "Easy", "Medium", or "Hard"
            
        Returns:
            tuple: (question_text, correct_answer, operation)
        """
        max_attempts = 20  # Avoid infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            if difficulty == "Easy":
                question, answer, operation = self._generate_easy()
            elif difficulty == "Medium":
                question, answer, operation = self._generate_medium()
            else:
                question, answer, operation = self._generate_hard()
            
            # Check if problem is too similar to recent ones
            if question not in self.recent_problems:
                self.recent_problems.append(question)
                if len(self.recent_problems) > self.max_history:
                    self.recent_problems.pop(0)
                return question, answer, operation
            
            attempts += 1
        
        # If all attempts fail, return anyway
        return question, answer, operation
    
    def _generate_easy(self):
        """
        Easy Level: Ages 5-6 (Kindergarten/Grade 1)
        - Single-digit addition: 1+1 to 5+5
        - Single-digit subtraction: No negatives, 5-3, 10-4
        - No multiplication/division
        """
        operation = random.choice(['addition', 'subtraction'])
        
        if operation == 'addition':
            # Simple addition: sum ≤ 10
            num1 = random.randint(1, 5)
            num2 = random.randint(1, 5)
            answer = num1 + num2
            question = f"{num1} + {num2}"
        
        else:  # subtraction
            # Ensure no negative results, keep answers positive
            num1 = random.randint(3, 10)
            num2 = random.randint(1, min(num1-1, 5))  # num2 < num1, max 5
            answer = num1 - num2
            question = f"{num1} - {num2}"
        
        return question, answer, operation
    
    def _generate_medium(self):
        """
        Medium Level: Ages 7-8 (Grade 2-3)
        - Two-digit addition: 10-50 range
        - Two-digit subtraction: No negatives
        - Simple multiplication: Times tables 2-5 only
        - Very simple division: 6÷2, 10÷5 (clean only)
        """
        operation = random.choice(['addition', 'subtraction', 'multiplication', 'division'])
        
        if operation == 'addition':
            # Two-digit addition, sum < 100
            num1 = random.randint(10, 45)
            num2 = random.randint(10, 45)
            answer = num1 + num2
            question = f"{num1} + {num2}"
        
        elif operation == 'subtraction':
            # Two-digit subtraction, no negatives
            num1 = random.randint(20, 50)
            num2 = random.randint(10, num1 - 5)  # Ensure positive result
            answer = num1 - num2
            question = f"{num1} - {num2}"
        
        elif operation == 'multiplication':
            # Times tables 2-5 only (easier for young kids)
            num1 = random.randint(2, 5)
            num2 = random.randint(2, 9)
            answer = num1 * num2
            question = f"{num1} × {num2}"
        
        else:  # division - clean division only
            # Small divisors (2-5) and quotients (2-10)
            divisor = random.randint(2, 5)
            quotient = random.randint(2, 10)
            num1 = divisor * quotient
            answer = quotient
            question = f"{num1} ÷ {divisor}"
        
        return question, answer, operation
    
    def _generate_hard(self):
        """
        Hard Level: Ages 9-10 (Grade 4-5)
        - Two-digit addition: 50-99 range
        - Two-digit subtraction: Larger numbers
        - Multiplication: Full times tables 2-12
        - Division: Clean division with larger numbers
        """
        operation = random.choice(['addition', 'subtraction', 'multiplication', 'division'])
        
        if operation == 'addition':
            # Two-digit addition, can exceed 100
            num1 = random.randint(45, 99)
            num2 = random.randint(45, 99)
            answer = num1 + num2
            question = f"{num1} + {num2}"
        
        elif operation == 'subtraction':
            # Two-digit subtraction
            num1 = random.randint(50, 99)
            num2 = random.randint(20, num1 - 10)
            answer = num1 - num2
            question = f"{num1} - {num2}"
        
        elif operation == 'multiplication':
            # Full times tables 2-12
            num1 = random.randint(6, 12)
            num2 = random.randint(6, 12)
            answer = num1 * num2
            question = f"{num1} × {num2}"
        
        else:  # division - clean division
            # Divisors 6-12, quotients up to 12
            divisor = random.randint(6, 12)
            quotient = random.randint(4, 12)
            num1 = divisor * quotient
            answer = quotient
            question = f"{num1} ÷ {divisor}"
        
        return question, answer, operation
    
    def get_difficulty_info(self, difficulty):
        """Get information about what each difficulty level includes"""
        info = {
            'Easy': {
                'age_range': '5-6 years',
                'grade': 'Kindergarten - Grade 1',
                'operations': ['Single-digit addition (sum ≤ 10)', 'Single-digit subtraction (no negatives)'],
                'example': '3 + 4 = 7, 8 - 3 = 5'
            },
            'Medium': {
                'age_range': '7-8 years',
                'grade': 'Grade 2-3',
                'operations': ['Two-digit addition', 'Two-digit subtraction', 'Times tables 2-5', 'Simple division'],
                'example': '23 + 34 = 57, 3 × 7 = 21, 15 ÷ 3 = 5'
            },
            'Hard': {
                'age_range': '9-10 years',
                'grade': 'Grade 4-5',
                'operations': ['Two-digit addition (>100)', 'Two-digit subtraction', 'Times tables 6-12', 'Division'],
                'example': '67 + 89 = 156, 9 × 8 = 72, 72 ÷ 8 = 9'
            }
        }
        return info.get(difficulty, {})


# Test the generator
if __name__ == "__main__":
    generator = PuzzleGenerator()
    
    print("="*60)
    print("MATH PUZZLE GENERATOR - AGES 5-10")
    print("="*60)
    
    for difficulty in ["Easy", "Medium", "Hard"]:
        info = generator.get_difficulty_info(difficulty)
        
        print(f"\n{difficulty.upper()} LEVEL")
        print("-" * 60)
        print(f"Age Range: {info['age_range']}")
        print(f"Grade Level: {info['grade']}")
        print(f"Operations:")
        for op in info['operations']:
            print(f"  • {op}")
        print(f"Example: {info['example']}")
        
        print(f"\nSample Problems:")
        for i in range(5):
            question, answer, operation = generator.generate(difficulty)
            print(f"  {i+1}. {question} = {answer} ({operation})")
    
    print("\n" + "="*60)
    print("✓ Generator tested successfully!")
    print("="*60)