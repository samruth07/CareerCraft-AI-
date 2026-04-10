"""
CareerCraft AI - GitHub Analyzer Tool
Fetches public repository data to verify candidate skills against actual code.
"""

import urllib.request
import json
import logging
import re

logger = logging.getLogger(__name__)

def analyze_github_profile(github_url: str) -> str:
    """
    Fetch public repositories and summarize languages/stats from a GitHub profile.
    
    Args:
        github_url: The candidate's GitHub URL.
        
    Returns:
        str: Summary of GitHub activity for the LLM context.
    """
    try:
        if not github_url or "github.com" not in github_url.lower():
            return "No valid GitHub URL provided."
            
        # Extract username from url
        match = re.search(r"github\.com/([^/\?]+)", github_url.lower())
        if not match:
            return "Could not extract GitHub username from URL."
            
        username = match.group(1).strip()
        
        api_url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=10"
        
        req = urllib.request.Request(api_url, headers={'User-Agent': 'CareerCraft-AI'})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                return f"GitHub API returned status {response.status}"
            data = json.loads(response.read().decode('utf-8'))
            
        if not data:
            return f"No public repositories found for user {username}."
            
        languages = {}
        stars = 0
        repo_details = []
        
        for repo in data:
            if repo.get('fork'):
                continue
                
            lang = repo.get('language')
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
                
            stars += repo.get('stargazers_count', 0)
            
            repo_details.append(f"- {repo.get('name')}: {repo.get('description', 'No description')} (Lang: {lang})")
            
        summary = f"GITHUB PROFILE REPORT ({username}):\n"
        summary += f"Total Stars on Recent Repos: {stars}\n"
        summary += f"Dominant Languages: {', '.join(k for k, v in sorted(languages.items(), key=lambda item: item[1], reverse=True)[:5])}\n"
        summary += "Recent Repositories:\n" + "\n".join(repo_details[:5])
        
        return summary
        
    except Exception as e:
        logger.error(f"GitHub Analysis failed: {str(e)}")
        return f"Failed to fetch GitHub data: {str(e)}"
