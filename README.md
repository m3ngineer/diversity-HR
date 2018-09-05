# Increasing gender equality in the workplace hiring process

Studies show that subtle gender-biased language in job advertisements can deter men and women from applying to certain jobs. Gender gaps are especially prevalent in insurance, business, finance, and tech sectors. This loss of diversity can negatively impact company performance. We’re here to increase diverse hiring by helping recruiters create gender-neutral job postings and establish inclusivity and diversity in the workplace.

As part of the Social Good Hackathon by Capgemini we sought to develop a technological solution to support increasing gender and cultural diversity in the workplace.

### Background
Research by [Gaucher et al](https://www.ncbi.nlm.nih.gov/pubmed/21381851) suggests that gendered wording in job descriptions and placement ads may affect female perception of company . These institutional mechanisms may prohibit women and cultural minorities from applying to particular jobs, which can perpetuate and sustain gender gaps in the workplace. 

Industries that are typically male-dominated may tend to have more masculine language (eg, dominant, competitive, leader). Experiments showed that altering a job description to include more masculine language increased the ratio of male:female applicants. Female applicants found these job positions less appealing.

### Solution
Job descriptions from the technology sector, a traditionally male-dominated industry, were scraped from Indeed. 
Using language identified as gender neutral, or gender specific by [Gaucher et al](https://www.ncbi.nlm.nih.gov/pubmed/21381851) we used job postings to train a bayesian model to determine whether a placement ad was gender-neutral or not.

### Upcoming functionality
The model will return a score and flag words that are considered gender-specific.

### To run the app:
This prototype is a web-based app built in python and Flask. To run, please use app.py.

<p align="center">
  <img src="/static/img/diversity-HR_landing.png" alt="Image of landing page" width="600"/>
</p>
The demo: https://pr.to/0PYECH/

### Contact
Please send any suggestions or comments to matthew.eng2@gmail.com.
