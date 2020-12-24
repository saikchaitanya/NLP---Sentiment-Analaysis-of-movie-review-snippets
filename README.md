# NLP---Sentiment-Analaysis-of-movie-review-snippets
Insights into BERT’s ability to recognize the sentiment of movie review snippets using transfer learning

# Executing the code
- As the flags already have predefined values, a single command will execute the file: "python test_saikrishna.py"
- the dataset (Huggingface NLP Tomatoes Sentiment problem) will  be automatically called inside the code itself

# Results and findings
- I am logging the loss values and accuracy values. These give us an idea of accuracy improves and loss decreases. Through this we can understand if our algorithm is working or not as ideally accuracy should increase with number of epochs and loss should decrease. 
- I did try to incorporate as many metrics as possible. I used the sklearn metrics instead. Also a lot of times accuracy might not give us a complete picture of the performance of our model, hence F-1 scores help.
- This code creates a new folder called experiments which stores the log files. 

