"""
Adversary Agent for CompliancePatchBench.
Generates new violations designed to evade the patcher agent.
Creates a self-improving adversarial loop.
"""

import random
from typing import Dict, List, Tuple


class AdversaryAgent:
    """
    Generates new compliance violations to challenge the patcher agent.
    Uses deterministic mutation strategies based on seed.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random = random.Random(seed)
        self.violation_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, List[Dict]]:
        """Load violation templates for different rule types."""
        return {
            "GDPR-ART5-1A": [
                {
                    "pattern": "logger.info(f\"User {user.email} action\")",
                    "description": "Logging email in f-string",
                    "difficulty": "easy",
                },
                {
                    "pattern": "logger.debug(f\"Processing {user.phone}\")",
                    "description": "Logging phone number",
                    "difficulty": "easy",
                },
                {
                    "pattern": "print(f\"User data: {user.email}, {user.address}\")",
                    "description": "Multiple PII in print",
                    "difficulty": "medium",
                },
                {
                    "pattern": "logger.info(f\"Request from {request.headers.get('X-User-Email')}\")",
                    "description": "Indirect PII logging",
                    "difficulty": "hard",
                },
            ],
            "GDPR-ART5-1C": [
                {
                    "pattern": "return jsonify({'user': user.__dict__})",
                    "description": "Exposing all user attributes",
                    "difficulty": "easy",
                },
                {
                    "pattern": "return {'password': user.password_hash, 'salt': user.salt}",
                    "description": "Exposing password components",
                    "difficulty": "medium",
                },
                {
                    "pattern": "response.set_cookie('session', user.session_token + user.password_hash)",
                    "description": "Password in cookie",
                    "difficulty": "hard",
                },
            ],
            "OWASP-A03": [
                {
                    "pattern": "db.execute(f\"SELECT * FROM users WHERE id={user_id}\")",
                    "description": "SQL injection via f-string",
                    "difficulty": "easy",
                },
                {
                    "pattern": "query = f\"UPDATE users SET name='{name}' WHERE id={id}\"",
                    "description": "SQL injection in UPDATE",
                    "difficulty": "medium",
                },
                {
                    "pattern": "cursor.execute(f\"DELETE FROM {table} WHERE id={id}\")",
                    "description": "Table name injection",
                    "difficulty": "hard",
                },
            ],
        }
    
    def generate_violation(
        self,
        rule_id: str,
        difficulty: str = "medium",
        context: Dict = None
    ) -> Tuple[str, Dict]:
        """
        Generate a new violation for the given rule.
        
        Args:
            rule_id: Rule to violate (e.g., "GDPR-ART5-1A")
            difficulty: "easy", "medium", or "hard"
            context: Optional context about what patcher has fixed
        
        Returns:
            (code_snippet, metadata)
        """
        templates = self.violation_templates.get(rule_id, [])
        if not templates:
            raise ValueError(f"Unknown rule_id: {rule_id}")
        
        # Filter by difficulty
        candidates = [t for t in templates if t["difficulty"] == difficulty]
        if not candidates:
            candidates = templates  # Fallback to all
        
        # Select template
        template = self.random.choice(candidates)
        
        # Add variation
        code = self._add_variation(template["pattern"], rule_id)
        
        metadata = {
            "rule_id": rule_id,
            "difficulty": template["difficulty"],
            "description": template["description"],
            "seed": self.seed,
        }
        
        return code, metadata
    
    def _add_variation(self, pattern: str, rule_id: str) -> str:
        """Add random variation to make violation less obvious."""
        variations = {
            "user.email": ["user.email", "current_user.email", "request_user.email"],
            "user.phone": ["user.phone", "user.phone_number", "user.mobile"],
            "password_hash": ["password_hash", "pwd_hash", "hashed_password"],
            "logger.info": ["logger.info", "logger.debug", "logger.warning"],
        }
        
        result = pattern
        for key, options in variations.items():
            if key in result:
                result = result.replace(key, self.random.choice(options))
        
        return result
    
    def mutate_fixed_code(
        self,
        fixed_code: str,
        original_violation: Dict
    ) -> Tuple[str, Dict]:
        """
        Mutate fixed code to reintroduce a similar violation.
        This creates an adaptive adversary that responds to patcher fixes.
        
        Args:
            fixed_code: Code that patcher fixed
            original_violation: Original violation that was fixed
        
        Returns:
            (mutated_code, new_violation_metadata)
        """
        rule_id = original_violation["rule_id"]
        
        # Strategy: Introduce violation in a different location
        # or use a different pattern for the same rule
        new_code, metadata = self.generate_violation(
            rule_id,
            difficulty="hard",  # Escalate difficulty
        )
        
        metadata["mutation_type"] = "adaptive"
        metadata["original_violation"] = original_violation
        
        return new_code, metadata
    
    def evaluate_patcher_performance(
        self,
        violations_fixed: int,
        total_violations: int,
        avg_reward: float
    ) -> str:
        """
        Determine next difficulty level based on patcher performance.
        
        Returns:
            "easy", "medium", or "hard"
        """
        fix_rate = violations_fixed / max(1, total_violations)
        
        if fix_rate >= 0.8 and avg_reward >= 2.0:
            return "hard"  # Patcher is strong, escalate
        elif fix_rate >= 0.5 and avg_reward >= 1.0:
            return "medium"  # Patcher is competent
        else:
            return "easy"  # Patcher is struggling, keep it simple
    
    def generate_curriculum(
        self,
        num_rounds: int = 5,
        initial_difficulty: str = "easy"
    ) -> List[Dict]:
        """
        Generate a curriculum of violations with increasing difficulty.
        
        Args:
            num_rounds: Number of training rounds
            initial_difficulty: Starting difficulty
        
        Returns:
            List of violation sets for each round
        """
        curriculum = []
        difficulty_progression = {
            "easy": "medium",
            "medium": "hard",
            "hard": "hard",
        }
        
        current_difficulty = initial_difficulty
        
        for round_num in range(num_rounds):
            round_violations = []
            
            # Generate 2-3 violations per round
            num_violations = self.random.randint(2, 3)
            
            for _ in range(num_violations):
                rule_id = self.random.choice(list(self.violation_templates.keys()))
                code, metadata = self.generate_violation(rule_id, current_difficulty)
                
                round_violations.append({
                    "code": code,
                    "metadata": metadata,
                    "round": round_num + 1,
                })
            
            curriculum.append({
                "round": round_num + 1,
                "difficulty": current_difficulty,
                "violations": round_violations,
            })
            
            # Escalate difficulty
            current_difficulty = difficulty_progression[current_difficulty]
        
        return curriculum


def demo_adversary():
    """Demonstrate adversary agent capabilities."""
    print("=" * 70)
    print("ADVERSARY AGENT DEMONSTRATION")
    print("=" * 70)
    print()
    
    adversary = AdversaryAgent(seed=42)
    
    # Demo 1: Generate violations
    print("1. GENERATE VIOLATIONS")
    print("-" * 70)
    for rule_id in ["GDPR-ART5-1A", "GDPR-ART5-1C", "OWASP-A03"]:
        code, metadata = adversary.generate_violation(rule_id, difficulty="medium")
        print(f"\nRule: {rule_id}")
        print(f"Code: {code}")
        print(f"Difficulty: {metadata['difficulty']}")
    print()
    
    # Demo 2: Adaptive mutation
    print("2. ADAPTIVE MUTATION")
    print("-" * 70)
    fixed_code = 'logger.info("User %s logged in", user.id)'
    original_violation = {"rule_id": "GDPR-ART5-1A", "line": 74}
    
    mutated_code, metadata = adversary.mutate_fixed_code(fixed_code, original_violation)
    print(f"Fixed Code: {fixed_code}")
    print(f"Mutated Code: {mutated_code}")
    print(f"Mutation Type: {metadata['mutation_type']}")
    print()
    
    # Demo 3: Curriculum generation
    print("3. CURRICULUM GENERATION")
    print("-" * 70)
    curriculum = adversary.generate_curriculum(num_rounds=3)
    for round_data in curriculum:
        print(f"\nRound {round_data['round']} (Difficulty: {round_data['difficulty']})")
        for v in round_data['violations']:
            print(f"  - {v['metadata']['rule_id']}: {v['metadata']['description']}")
    print()
    
    # Demo 4: Difficulty adaptation
    print("4. DIFFICULTY ADAPTATION")
    print("-" * 70)
    scenarios = [
        (1, 3, 0.5, "Struggling patcher"),
        (2, 3, 1.5, "Competent patcher"),
        (3, 3, 2.5, "Strong patcher"),
    ]
    
    for fixed, total, reward, desc in scenarios:
        difficulty = adversary.evaluate_patcher_performance(fixed, total, reward)
        print(f"{desc}: {fixed}/{total} fixed, reward={reward:.1f} -> {difficulty}")
    print()
    
    print("=" * 70)
    print("ADVERSARY READY FOR SELF-PLAY")
    print("=" * 70)


if __name__ == "__main__":
    demo_adversary()
