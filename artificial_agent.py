import deep_nlp_module as dnm
import nltk

if __name__ == "__main__":
    """
    Main Program
    """
    chatbot_classifier, resumes_classifier, chatbot_most_common_words, resumes_most_common_words = dnm.extract_module();
    
    end = "";
    while (end != "Yes"):
        test = input("BOT: Hello! Would you like some therapy? Or would you like to submit your resume? (1 - Therapy, 2 - Resume): ");
        
        if (test == "Therapy"):
            response = input("BOT: Tell me what is bothering you?\n");
            response = nltk.word_tokenize(response);
            response = dnm.extract_features(response, chatbot_most_common_words);
            chatbot_reply = chatbot_classifier.classify(response);
            if(chatbot_reply == "flagged"):
                print("BOT: Looks like you are going through a hard time! I suggest you get some help.");
            else:
                print("BOT: From my experience, you will be okay. Just stay strong!");
                
        elif (test == "Resume"):
            response = input("BOT: Please submit your resume in simple text form!\n");
            response = nltk.word_tokenize(response);
            response = dnm.extract_features(response, chatbot_most_common_words);
            chatbot_reply = chatbot_classifier.classify(response);
            if(chatbot_reply == "flagged"):
                print("BOT: Awesome resume! Hang tight, we will contact you soon for an interview.");
            else:
                print("BOT: Unfortunately, we are not looking to hire someone with your qualifications yet :( Keep your eyes open for new postings!");
                
        else:
            print("Sorry! I cannot understand what you said. I only work in therapy and handling resumes. Make sure you've got no typos!");
        
        end = input("Do you want to end the chat? (Yes/No): ");
        
        if (end != "Yes" and end != "No"):
            print("Sorry! You can only answer by Yes or No. Make sure you've got no typos!");
            end = input("Do you want to end the chat? (Yes/No): ");