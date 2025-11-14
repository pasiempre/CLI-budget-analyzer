Project Name: Expense & Budget Analyzer

Video Demo: https://youtu.be/HqwP4EQVKvY

Description:

This project came to fruition due to my personal need to keep my finances in order as I currently dont have a job. Every expense matters, and what better way to assist in that than to program something with Python to keep me in line!

The CS50 curriculum was very detailed but there were areas that I wanted to dig into more. Below, I will list books that I used to assist with my thought process and knowledge to build out this project.

1. Python for Data Analysis - Wes Mckinney
2. Fluent Python - Ramalho
3. Data Science for Business - Provost & Fawcett
4. Database System Concepts - Abraham Silberschatz

Via the knowledge acquired from these books and the CS50 curriculum i decided to implement a CLI mode that is fully interactive for users to go through. You are meant to upload your finance document and my program identifies this along with a budget CSV if you decide to upload one.

The data is then cleaned and normalized to fit the structure of the program which means also aggregating by category and computing the income / expense by month.

An additional feature added was budget alerts - which is a part of the CLI commands that will show if you are over budget in any category.

The interactive CLI provides you with an option for:

[s] Summary
[t] Category Totals
[o] Top 10 Bar
[i] Income / Expense / Net by Month
[b] Budget Alerts
[q] Quit

Each of these options has a function within project.py to prompt the correct information to show. The top 10 bar is essentially your top 10 spending categories visualized within the CLI with hash symbols creating a bar graph.

Those functions also have corresponding pytest tests to ensure that they wont break under different conditions.

This was a very helpful project because it introduced me to information about pandas and numpy, regex, argparse, pathlib, glob, os, CLI design and further testing with Pytest. All of which I needed more experience with.

I have also subscribed and paid for additional CS50 courses through HarvardX and EdX to hopefully strengthen my skills in these different areas.

This project put everything together for me and was an excellent opportunity to put this knowledge to use. I hope to be able to look back at it in the future and remember the struggles of putting it together and the valuable insight it provided. I have committed to learning about data science and python for life, so definitely excited for more of this to come.
