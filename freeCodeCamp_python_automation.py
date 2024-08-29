# Automate with Python - Full Course for Beginners
# https://www.youtube.com/watch?v=s8XjEuplx_U
# ~ 3 hours

# Time-Stamps
# (0:00:00) Introduction
# (0:00:29) Hacker News Headlines Emailer - Tutorial 1
# (0:01:13) Introduction to Web Scraping
# (0:03:08) Setting up the Environment
# (0:06:30) Project Script
# (0:11:00) Website Structure of Hacker News FrontPage
# (0:21:00) Sending Email from Python
# (0:35:15) Building the Headlines Email Module
# (0:39:07) TED Talk Downloader - Tutorial 2
# (0:39:49) Installation and Introduction to requests package
# (0:41:25) Installation and Introduction to BeautifulSoup
# (0:43:25) Building the basic script to download the video
# (0:49:37) Generalising the Script to get Arguments
# (0:53:49) Table Extractor from PDF - Tutorial 3  
# (0:54:44) Basics of PDF Format
# (0:57:05) Installing required Python Modules
# (1:02:16) Extracting Table from PDF
# (1:06:51) Quick Introduction to Jupyter Notebook
# (1:08:29) PDF Extraction on Jupyter Notebook
# (1:15:29) Pandas and Write Table as CSV Excel
# (1:21:02) Automated Bulk Resume Parser - Tutorial 4
# (1:22:15) Different Formats of Resumes and marking relevant Information
# (1:25:50) Project Architecture and Brief Overview of the required packages and installations
# (1:34:48) Basics of Regular Expression in Python
# (1:41:38) Basic Overview of Spacy Functions
# (1:49:55) Extracting Relevant Information from the Resumes
# (2:16:46) Completing the script to make it a one-click CLI
# (2:28:09) Image Type Converter - Tutorial 5
# (2:29:09) Different type of Image Formats
# (2:31:33) What is an Image type convertor
# (2:33:04) Introduction to Image Manipulation in Python
# (2:34:51) Building an Image type converting Script
# (2:40:03) Converting the script into a CLI Tool
# (2:44:27) Building an Automated News Summarizer - Tutorial 6
# (2:46:27) What is Text Summarization
# (2:47:46) Installing Gensim and other Python Modules
# (2:52:43) Extracting the required News Source
# (2:59:38) Building the News Summarizer
# (3:07:14) Scheduling the News Summarizer
# (3:10:25) Thank you



# (0:00:29) Hacker News Headlines Emailer - Tutorial 1

# Steps
# 1. Get the front page
# 2. Scrape the content (title/link)
# 3. Build email body/content
# 4. Authenticate the email
 

# (0:03:08) Setting up the Environment

# MODULES
# requests - for http requests
# bs4 - "beautiful soup" for web scraping 
# smtplib - for email authentication  [Available by default]
# email.mime - for creation of the email body [Available by default]
# datetime - for accessing and manipulating date and time [Available by default]
 
# INSTALLING EXTERNAL MODULES
# in the terminal...
    pip (or pip3) install <module-name>
# 
# in this case...
    pip install requests
    pip3 install beautifulsoup4
# the rest are already installed, so we just need to import that
    import smtplib
    import email.mime
    import datetime


# (0:06:30) Project Script

import requests  # http requests
from bs4 import BeautifulSoup  # web scraping 
import smtplib  # send the email
from email.mime.multipart import MIMEMultipart  # email body
from email.mime.text import MIMEText
import datetime  # system date and time manipulation 

now = datetime.datetime.now()  # get the current date and time for the email subject line

content = ''  # creating an empty string

# Extracting Hacker News Stories
def extract_news(url):  # create a new function that will accept a url
    print('Extracting Hacker News Stories...')  # let us know it has started
    temp_content = ''
    temp_content += ('<b>HN Top Stories:</b>\n' + '<br>' + '-'*50 + '<br>')
    response = requests.get(url)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')

    for i, tag in enumerate(soup.find_all('td', attrs = {'class':'title','valign':''})):
        temp_content += ((str(i+1) + ' :: ' + tag.text + '\n' + '<br>') if tag.text != 'More' else '')
        # print(tag.prettify) # find_all('span', attrs = {'class':'sitestr'}))
    return(temp_content)

# pick up at 00:08:59





