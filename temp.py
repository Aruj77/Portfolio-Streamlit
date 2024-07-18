    with st.container():
        text_column, image_column = st.columns((2,1))
        with text_column:
            st.subheader("Data Scraping Project 1")
            st.write("*It is a Data Sraping project of a website that sells products based on Yoga items.*")
            st.markdown("""
            - All the data of 64 pages is taken.
            - Used Libraries: `Numpy`, `Pandas`, `Matplotlib`, `csv`, `re`, `etc`.
            """)
            st.write("[Github Repo](https://github.com/Aruj77/Data-Scraping-Pro-1)")
        with image_column:
            st.image(images_projects[0], width=350)
    with st.container():
        text_column, image_column = st.columns((2,1))
        with text_column:
            st.subheader("Loan Approval")
            st.write(
                "*This project aims to develop a machine learning model that predicts loan approval decisions based on applicant data. By analyzing various features such as income, credit history, loan amount, and more, the model helps financial institutions make informed decisions regarding loan approvals.*"
            )
            st.markdown("""
            - Features: Data Processing, EDA, Feature Engineering, Model Building, Model Evaluation, Prediction
            - Used Technology: `Python`, `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `Jupyter Notebook`
            """)
            st.write("[Github Repo](https://github.com/Aruj77/Loan-Approval)")
        with image_column:
            st.image(images_projects[1])
    with st.container():
        text_column, image_column = st.columns((2,1))
        with text_column:
            st.subheader("Remove Outliers in a DataSet")
            st.write("*In this project you will find a function which will help you in cleaning data by removing all the unwanted outliers.*")
            st.markdown("""
            - Outliers - These are the values that are not needed in the dataset as the are very far away from the range.
            - For finding these outliers we need to find: q1 = 25%tile q3 = 75%tile iqr = q3-q1 lower fence = q1 - 1.5iqr upper fence = q3 + 1.5iqr
            """)
            st.write("[Github Repo](https://github.com/Aruj77/Remove-Outliers)")
        with image_column:
            st.image(images_projects[2])
    with st.container():
        text_column, image_column = st.columns((2,1))
        with text_column:
            st.subheader("Car and Pedestrian Detection")
            st.write(
                "*In this project the code is using harcascade model of machine learning to detect cars and pedestrian.*"
            )
            st.markdown(
                """
            - Haar cascade is an algorithm that can detect objects in images, irrespective of their scale in image and location.
            - Haar cascade uses the cascading window, and it tries to compute features in every window and classify whether it could be an object. For more details on its working, refer to this link.
            - Used Libraries: `opencv-python`, `numpy`, `haar Cascade`
            """
            )
            st.write(
                "[Github Repo](https://github.com/Aruj77/Car-and-Pedestrian-Detection-Machine-Learning)"
            )

        with image_column:
            st.image(images_projects[3])
    with st.container():
        text_column, image_column = st.columns((2,1))
        with text_column:
            st.subheader("Face Detection Using Machine Learning")
            st.write(
                "*This project focuses on developing a face detection system using various computer vision and machine learning techniques. The system can detect faces in images and videos, and can be used for applications such as security, photo organization, and more.*"
            )
            st.markdown(
                """
            - Features: Image preprocessing, face detection, drawing bounding boxes, real-time detection.
            - The project structure includes directories for data storage (input images and videos), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for face detection and processing, models, and a requirements file listing the required packages.
            - Used Libraries: `OpenCV`, `Dlib`, `NumPy`, `Matplotlib`.
            """
            )
            st.write("[Github Repo](https://github.com/Aruj77/Face-Detection)")

        with image_column:
            st.image(images_projects[4])
    with st.container():
        text_column, image_column = st.columns((2,1))
        with text_column:
            st.subheader("Hand Detection Using Machine Learning")
            st.write("*This project focuses on developing a hand detection system using various computer vision and machine learning techniques. The system can detect hands in images and videos, and can be used for applications such as gesture recognition, human-computer interaction, and more.*")
            st.markdown(
                """
            - The project structure includes directories for data storage (input images and videos), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for hand detection and processing, models, and a requirements file listing the required packages.
            - Features: Image preprocessing, hand detection, drawing bounding boxes, real-time detection.
            - Used Libraries: `OpenCV`, `Dlib`, `NumPy`, `Matplotlib`
            """
            )
            st.write("[GitHub Repo](https://github.com/Aruj77/Hand-Detection-Machine-Learning)")

        with image_column:
            st.image(images_projects[5])
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Diabities Prediction Using Machine Learning")
            st.write(
                "*This project focuses on developing a diabetes prediction system using various machine learning techniques. The system aims to predict the likelihood of diabetes in individuals based on their medical data, helping in early diagnosis and management of the disease.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage (input datasets), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, training, and evaluating models, models, and a requirements file listing the required packages.
            - Features: Data preprocessing, feature selection, model training, model evaluation, prediction.
            - Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/Diabeties-Prediction-Machine-Learning)"
            )

        with image_column:
            st.image(images_projects[6])
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Auction Analysis Using Machine Learning")
            st.write(
                "*This project focuses on analyzing auction data using various data analysis and visualization techniques. The goal is to uncover patterns, trends, and insights that can inform decision-making processes related to auctions.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage (input datasets), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing and analyzing data, models, and a requirements file listing the required packages.
            - Features: Data preprocessing, exploratory data analysis, data visualization, statistical analysis, trend analysis.
            - Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `Jupyter NoteBook`
            """
            )
            st.write("[GitHub Repo](https://github.com/Aruj77/Auction-Analysis)")

        with image_column:
            st.image(images_projects[7], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Handwritind Detection Using Machine Learning")
            st.write(
                "*This project focuses on developing a handwriting detection system using various computer vision and machine learning techniques. The system aims to recognize and interpret handwritten text from images, which can be used for applications such as document digitization, note-taking, and more.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage (input images), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, detecting, and recognizing handwriting, models, and a requirements file listing the required packages.
            - Features: Image preprocessing, text detection, character recognition, text extraction, real-time detection.
            - Used Libraries: `OpenCV`, `TensorFlow` or `PyTorch`, `NumPy`, `Matplotlib`, `Tesseract OCR`, `Pillow`
            """
            )
            st.write("[GitHub Repo](https://github.com/Aruj77/Handwriting-Detection)")

        with image_column:
            st.image(images_projects[8], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Function to Find the Score of a Machine Learning Model")
            st.write(
                "*This project provides a function to evaluate the performance of a machine learning model. It includes various metrics such as accuracy, precision, recall, F1 score, and more, to help understand how well the model is performing.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage (input datasets), Jupyter Notebooks for model training and evaluation, Python scripts for evaluating model performance, models, and a requirements file listing the required packages.
            - Features: Model training, model evaluation, performance metrics calculation, visualization of results.
            - Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`
            """
            )
            st.write("[GitHub Repo](https://github.com/Aruj77/Function-to-Find-the-Score-of-a-Machine-Learning-Model)")

        with image_column:
            st.image(images_projects[9], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("All Machine Learning Model Score Prediction")
            st.write(
                "*This project leverages LazyPredict to quickly evaluate and compare the performance of multiple machine learning models on a given dataset. LazyPredict provides a simple way to run a variety of models and get their scores without extensive coding.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage (input datasets), Jupyter Notebooks for running LazyPredict, Python scripts for evaluating model performance, models, and a requirements file listing the required packages.
            - Features: Model training, model evaluation, performance metrics calculation, comparison of multiple models.
            - Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `LazyPredict`
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/All-ML-Model-Score-Prediction)"
            )

        with image_column:
            st.image(images_projects[10], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Object Detection")
            st.write(
                "*This project focuses on developing an object detection system using the YOLO (You Only Look Once) algorithm. The system aims to detect and classify objects in images and videos in real-time, which can be used for applications such as surveillance, autonomous driving, and more.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage (input images and videos), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing and detecting objects, models, and a requirements file listing the required packages.
            - Features: Image preprocessing, object detection, bounding boxes, real-time detection.
            - Used Libraries: `OpenCV`, `TensorFlow` or `PyTorch`, `NumPy`, `Matplotlib`, `YOLOv3` or `YOLOv4`
            """
            )
            st.write("[GitHub Repo](https://github.com/Aruj77/Object-Detection)")

        with image_column:
            st.image(images_projects[11], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Wabra Chat Analyzer")
            st.write(
                "*This project focuses on analyzing chat data to extract sentiments and provide analytical insights. The system aims to identify the sentiment (positive, negative, neutral) of chat messages and generate various analytical reports, which can be used for customer service improvement, user behavior analysis, and more.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage (input chat logs), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, analyzing, and visualizing data, models, and a requirements file listing the required packages.
            - Features: Data preprocessing, sentiment analysis, analytical reporting, data visualization.
            - Used Libraries: `Pandas`, `NumPy`, `NLTK`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `TextBlob`, `VADER Sentiment Analysis`, `Jupyter Notebook`.
            """
            )
            st.write("[GitHub Repo](https://github.com/Aruj77/Wabra-Chat-Analyzer)", " | ","[Live Demo](https://wabra-chat-analyzer.onrender.com/)")

        with image_column:
            st.image(images_projects[12], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Fitness Club")
            st.write(
                "*This project focuses on developing a responsive and interactive gym website using React. The website aims to provide information about the gym, its services, trainers, membership plans, and more, while also allowing users to book classes and contact the gym.*"
            )
            st.markdown(
                """
            - The project structure includes directories for components (React components), pages (different pages of the website), services (API calls and services), assets (images, icons, styles), and a configuration file listing the required packages.
            - Features: Responsive design, dynamic content rendering, booking system, contact form, integration with external APIs.
            - Used Libraries: `Node.js`, `npm` or `yarn`, `React`, `React Router`, `Redux`, `Axios`, Styled Components or `CSS/SASS`, `Formik` for forms, `Yup` for form validation.
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/Fitness-Club-)",
                " | ",
                "[Live Demo](https://fitnessone.netlify.app/)",
            )

        with image_column:
            st.image(images_projects[13], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Chat App")
            st.write(
                "*This project focuses on developing a real-time chat application using Flutter. The app aims to provide a seamless and interactive messaging experience with features such as user authentication, real-time messaging, media sharing, and more.*"
            )
            st.markdown(
                """
            - The project structure includes directories for UI components (widgets), screens (different pages of the app), services (Firebase services), models (data models), and a configuration file listing the required packages.
            - Features: User authentication, real-time messaging, media sharing (images, videos), push notifications, user profiles, chat rooms/groups.
            - Used Libraries: `Flutter`, `Dart`, `Firebase` (Firestore, Authentication, Storage), `Provider` or `Bloc` for state management.
            """
            )
            st.write("[GitHub Repo](https://github.com/Aruj77/Chat-App)")

        with image_column:
            st.image(images_projects[14], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Weather App")
            st.write(
                "*This project focuses on developing a weather application using React. The app aims to provide real-time weather information, forecasts, and weather-related alerts for various locations.*"
            )
            st.markdown(
                """
            - The project structure includes directories for components (React components), pages (different pages of the application), services (API calls), assets (images, icons, styles), and a configuration file listing the required packages.
            - Features: Real-time weather updates, weather forecasts, location-based weather, search functionality, weather alerts, responsive design.
            - Used Libraries: `Node.js`, `npm` or `yarn`, `React`, `Axios` (for API calls), `React Router` (for navigation), Styled Components or `CSS/SASS`, `OpenWeatherMap API`, `Tailwind`
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/weather-app-react)",
                " | ",
                "[Live Demo](https://weatherapp-4d642d49z-aruj77.vercel.app/)",
            )

        with image_column:
            st.image(images_projects[15], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Reidentification in CCTV ")
            st.write(
                "*This project focuses on developing a system for reidentifying individuals in CCTV footage using the YOLO (You Only Look Once) object detection algorithm. The goal is to accurately detect and track individuals across multiple camera feeds for applications such as security, surveillance, and forensic analysis.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage (input videos and images), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, detecting, and reidentifying individuals, models, and a requirements file listing the required packages.
            - Features: Object detection, reidentification, multi-camera tracking, real-time processing, data visualization.
            - Used Libraries: `OpenCV`, `TensorFlow` or `PyTorch`, `NumPy`, `Matplotlib`, `YOLOv3` or `YOLOv4` weights and configuration files, `Scikit-learn`
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/Reidentification-CCTV-footage-using-yolo)"
            )

        with image_column:
            st.image(images_projects[16], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Person, Gun, and Number Plate Detection ")
            st.write(
                "*This project focuses on developing a detection system using the YOLO (You Only Look Once) algorithm to identify persons, guns, and number plates in images and videos. The goal is to provide a robust solution for security and surveillance applications.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage (input videos and images), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, detecting, and analyzing data, models, and a requirements file listing the required packages.
            - Features: Object detection for persons, guns, and number plates, real-time processing, bounding boxes, alert generation.
            - Used Libraries: `OpenCV`, `TensorFlow` or `PyTorch`, `NumPy`, `Matplotlib`, `YOLOv3` or `YOLOv4` weights and configuration files, `Scikit-learn`
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/Yolo-model-person-gun-and-number-plate-detection-)"
            )

        with image_column:
            st.image(images_projects[17], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("poRt")
            st.write(
                "*This project focuses on developing a responsive and user-friendly website for selling second-hand laptops. The website provides features for listing laptops, searching for laptops, user authentication, and secure transactions.*"
            )
            st.markdown(
                """
            - The project structure includes separate directories for frontend and backend code, each containing components, pages, services, models, routes, and configuration files.
            - Features: User authentication, laptop listings, search functionality, filtering options, detailed laptop descriptions, user profiles, secure transactions, responsive design.
            - Used Libraries: `Node.js`, `npm` or `yarn`, `React`, `Redux` (for state management), `Axios` (for API calls), `React Router` (for navigation), Styled Components or `CSS/SASS`, `Express` (for backend), `MongoDB` (for database).
            """
            )
            st.write("[GitHub Repo](https://github.com/Aruj77/poRt)")

        with image_column:
            st.image(images_projects[18], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Urine Strip Analyzer")
            st.write(
                "*This project focuses on developing a Urine Strip Analyzer application that uses image processing to analyze urine strips. The backend is developed using Django, the frontend uses HTML, CSS, and JavaScript (jQuery), and image processing is done using OpenCV. The database used is SQLite.*"
            )
            st.markdown(
                """
            - The project structure includes directories for the backend (Django), frontend (HTML/CSS/JavaScript), image processing (OpenCV scripts), and database (SQLite).
            - Features: User authentication, urine strip image upload, image processing for analysis, results display, user profile management, historical data storage.
            - Used Libraries: `Django`, `OpenCV`, `SQLite`, `HTML`, `CSS`, and `JavaScript` (jQuery).
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/Urine-Strip-Analyzer) | [Live Demo](https://urine-strip-analyzer.vercel.app/)"
            )

        with image_column:
            st.image(images_projects[19], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Course List and Details")
            st.write(
                "*This project focuses on developing a responsive and user-friendly website for listing courses and displaying their details. The frontend is developed using React, and the website provides features such as course listings, search functionality, filtering options, and detailed course descriptions.*"
            )
            st.markdown(
                """
            - The project structure includes directories for components (React components), pages (different pages of the application), services (API calls), assets (images, icons, styles), and a configuration file listing the required packages.
            - Features: User authentication, course listings, search functionality, filtering options, detailed course descriptions, user profiles, responsive design.
            - Used Libraries: `Node.js`, `npm` or `yarn`, `React`, `Redux` (for state management), `Axios` (for API calls), `React Router` (for navigation), Styled Components or `CSS/SASS`
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/Course-List-and-details-React-) | [Live Demo](https://course-list-and-details-react.vercel.app/)"
            )

        with image_column:
            st.image(images_projects[20], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Fraudulent Data Analysis")
            st.write(
                "*This project focuses on analyzing data to detect fraudulent activities using various data science and machine learning techniques. The aim is to develop a system that can identify anomalies and patterns indicative of fraud.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage, Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, feature engineering, model training, and evaluation, and a requirements file listing the required packages.
            - Features: Data preprocessing, feature engineering, exploratory data analysis, model training and evaluation, anomaly detection, visualization of results.
            - Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `TensorFlow` or `PyTorch`, `Jupyter Notebook`
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/Fraudulent-Data-Analysis)"
            )

        with image_column:
            st.image(images_projects[21], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("Stock Monitoring Platform")
            st.write(
                "*This project focuses on developing a Stock Monitoring Platform that allows users to track and analyze stock market data. The platform provides real-time stock updates, historical data analysis, visualizations, and alerts for significant market events.*"
            )
            st.markdown(
                """
            - The project structure includes directories for data storage, Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, feature engineering, model training, and evaluation, and a requirements file listing the required packages.
            - Features: Real-time stock data updates, historical data analysis, stock performance visualization, user-defined alerts, portfolio tracking, user authentication.
            - Used Libraries: `Django`, `Pandas`, `NumPy`, `Matplotlib`, `Plotly`, `Requests`, `SQLite` (default for Django), `React.js`, `CSS`, `JavaScript` (jQuery)
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/Stock-Monitoring-Platform)"
            )

        with image_column:
            st.image(images_projects[22], width=350)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("PixelCops: Image Authentication and Origin Tracing")
            st.write(
                "*In today's digital age, distinguishing authentic images from those generated by AI, especially in sensitive areas like news and social media, is increasingly challenging. PixelCops addresses this crucial need by offering a reliable solution for verifying and tracing the origin of digital content. This system analyzes images pixel by pixel to detect manipulations and identify the source of the images.*"
            )
            st.markdown(
                """
            - The project structure includes directories for the backend (Django), frontend (HTML/CSS/JavaScript), image analysis scripts (OpenCV and machine learning models), and database (SQLite).
            - Features: Pixel-by-pixel image analysis, AI-generated image detection, image manipulation detection, origin tracing, user authentication, detailed reports, responsive design.
            - Used Libraries: `Solidity`, `Rust`, `React Native`, `TypeScript`, `React.js`, `RollUp`, `RISC0`, `ZKVM`
            """
            )
            st.write(
                "[GitHub Repo](https://github.com/Aruj77/PixelCops)"
            )

        with image_column:
            st.image(images_projects[23], width=150)
    with st.container():
        text_column, image_column = st.columns((2, 1))
        with text_column:
            st.subheader("GPT Clone")
            st.write(
                "*This project aims to create a conversational AI system similar to OpenAI's ChatGPT. The system will use natural language processing techniques to understand and generate human-like responses in text-based conversations.*"
            )
            st.markdown(
                """
            - Features: Natural language understanding (NLU) for processing user inputs, Response generation using machine learning models, Context management to maintain coherent conversations, User authentication and personalized responses, Integration with messaging platforms
            - Used Libraries: `PyTorch`, `Transformers library` (for pre-trained language models), Flask (for API endpoints), `React.js`, `CSS`, `JavaScript` (for frontend if applicable), `SQLite` or `MongoDB` (for storing conversation history and context).
            """
            )
            st.write("[GitHub Repo](https://github.com/Aruj77/GPT-Clone)")

        with image_column:
            st.image(images_projects[24], width=350)
