import requests
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GithubIssuesCollector:
    def __init__(self, github_token: str):
        self.token = github_token
        self.headers = {"Authorization": f"token {self.token}"}

    def fetch_comments(self, repo: str, issue_number: int) -> List[str]:
        """Fetch all comments for a given issue."""
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            return []
        return [comment["body"] for comment in response.json()]

    def fetch_issues(self, repo: str, max_issues: int = 20) -> List[Dict[str, Any]]:
        """Fetch GitHub issues with their comments."""
        logger.info(f"Fetching up to {max_issues} issues from {repo}")
        
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": "closed", "per_page": max_issues}
        
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            logger.error(f"Failed to fetch issues: {response.status_code}")
            return []

        issues_data = []
        for issue in response.json():
            comments = self.fetch_comments(repo, issue["number"])
            
            issues_data.append({
                "id": issue["id"],
                "title": issue["title"],
                "body": issue["body"],
                "comments": comments,
                "fix_info": self._extract_fix_info(issue["body"], comments)
            })

        logger.info(f"Fetched {len(issues_data)} issues")
        return issues_data

    def _extract_fix_info(self, body: str, comments: List[str]) -> List[str]:
        """Extract fix-related information from issue body and comments."""
        fix_info = []
        if body:
            fix_info.append(body)
        fix_info.extend(comments)
        return fix_info