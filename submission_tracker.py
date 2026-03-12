import pandas as pd
import json
from datetime import datetime
import os

class SubmissionTracker:
    def __init__(self):
        self.tracker_file = "submission_log.json"
        self.load_log()
    
    def load_log(self):
        """Load existing submission log"""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = {
                "submissions": [],
                "next_submission_number": 1,
                "total_submissions": 0,
                "daily_submissions": 0,
                "last_submission_date": None
            }
    
    def save_log(self):
        """Save submission log"""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.log, f, indent=2)
    
    def add_submission(self, filename, model_type, cv_score, description, expected_lb_score=None):
        """Add a new submission to the log"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Reset daily count if new day
        if self.log["last_submission_date"] != today:
            self.log["daily_submissions"] = 0
        
        submission = {
            "submission_number": self.log["next_submission_number"],
            "filename": filename,
            "model_type": model_type,
            "cv_score": cv_score,
            "expected_lb_score": expected_lb_score,
            "description": description,
            "date": today,
            "time": datetime.now().strftime("%H:%M:%S"),
            "daily_submission": self.log["daily_submissions"] + 1,
            "status": "pending"  # pending, submitted, scored
        }
        
        self.log["submissions"].append(submission)
        self.log["next_submission_number"] += 1
        self.log["total_submissions"] += 1
        self.log["daily_submissions"] += 1
        self.log["last_submission_date"] = today
        
        self.save_log()
        return submission
    
    def get_submission_plan(self):
        """Get recommended submission plan"""
        plan = [
            {
                "step": 1,
                "filename": "submission_baseline.csv",
                "script": "train_and_predict.py",
                "model": "XGBoost (default)",
                "description": "Baseline submission with default parameters",
                "priority": "HIGH",
                "cv_expectation": "0.45-0.50",
                "notes": "Submit first to get baseline score"
            },
            {
                "step": 2,
                "filename": "submission_optimized.csv", 
                "script": "hyperparameter_optimization.py",
                "model": "XGBoost (optimized)",
                "description": "Hyperparameter-tuned XGBoost",
                "priority": "HIGH",
                "cv_expectation": "0.50-0.55",
                "notes": "Submit after reviewing baseline results"
            },
            {
                "step": 3,
                "filename": "submission_ensemble.csv",
                "script": "train_and_predict.py (modified)",
                "model": "Weighted Ensemble",
                "description": "Ensemble of top 3 models",
                "priority": "MEDIUM",
                "cv_expectation": "0.52-0.58",
                "notes": "Submit if individual models perform well"
            },
            {
                "step": 4,
                "filename": "submission_advanced.csv",
                "script": "train_and_predict.py (with advanced features)",
                "model": "Advanced XGBoost",
                "description": "With additional feature engineering",
                "priority": "LOW",
                "cv_expectation": "0.55-0.62",
                "notes": "Submit only if previous submissions show promise"
            }
        ]
        return plan
    
    def get_current_status(self):
        """Get current submission status"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        status = {
            "total_submissions": self.log["total_submissions"],
            "remaining_submissions": 200 - self.log["total_submissions"],
            "today_submissions": self.log["daily_submissions"] if self.log["last_submission_date"] == today else 0,
            "remaining_today": 10 - (self.log["daily_submissions"] if self.log["last_submission_date"] == today else 0),
            "last_submission": self.log["submissions"][-1] if self.log["submissions"] else None
        }
        
        return status
    
    def print_submission_plan(self):
        """Print the recommended submission plan"""
        plan = self.get_submission_plan()
        status = self.get_current_status()
        
        print("🎯 FINANCIAL HEALTH PREDICTION - SUBMISSION PLAN")
        print("=" * 60)
        
        print(f"\n📊 CURRENT STATUS:")
        print(f"  Total submissions made: {status['total_submissions']}/200")
        print(f"  Submissions today: {status['today_submissions']}/10")
        print(f"  Remaining today: {status['remaining_today']}")
        print(f"  Remaining total: {status['remaining_submissions']}")
        
        if status['last_submission']:
            print(f"\n📝 LAST SUBMISSION:")
            print(f"  File: {status['last_submission']['filename']}")
            print(f"  Model: {status['last_submission']['model_type']}")
            print(f"  CV Score: {status['last_submission']['cv_score']}")
            print(f"  Date: {status['last_submission']['date']}")
        
        print(f"\n🚀 RECOMMENDED SUBMISSION STRATEGY:")
        print("-" * 60)
        
        for i, step in enumerate(plan, 1):
            status_icon = "✅" if i <= self.log["total_submissions"] else "🔄" if i == self.log["total_submissions"] + 1 else "⏳"
            print(f"\n{status_icon} STEP {step['step']}: {step['filename']}")
            print(f"   Script: {step['script']}")
            print(f"   Model: {step['model']}")
            print(f"   Description: {step['description']}")
            print(f"   Priority: {step['priority']}")
            print(f"   Expected CV: {step['cv_expectation']}")
            print(f"   Notes: {step['notes']}")
        
        print(f"\n💡 TODAY'S RECOMMENDATION:")
        if status['remaining_today'] == 0:
            print("  ⚠️  NO SUBMISSIONS REMAINING TODAY!")
            print("  💤 Wait until tomorrow to submit again.")
        elif status['total_submissions'] == 0:
            print("  🎯 SUBMIT STEP 1: Run 'python train_and_predict.py'")
            print("  📁 File to submit: submission_baseline.csv")
            print("  💬 Comments: Baseline XGBoost model with default parameters")
        elif status['total_submissions'] == 1:
            print("  🎯 SUBMIT STEP 2: Run 'python hyperparameter_optimization.py'")
            print("  📁 File to submit: submission_optimized.csv") 
            print("  💬 Comments: Hyperparameter-tuned XGBoost - expect improvement")
        else:
            print(f"  🎯 Continue with STEP {status['total_submissions'] + 1}")
            print("  📁 Check the plan above for next submission")
        
        return plan, status

def main():
    """Main function to display submission plan"""
    tracker = SubmissionTracker()
    plan, status = tracker.print_submission_plan()
    
    # Save the plan for reference
    tracker.save_log()
    
    return tracker, plan, status

if __name__ == "__main__":
    tracker, plan, status = main()
