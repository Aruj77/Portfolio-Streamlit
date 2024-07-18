from PIL import Image
image_names_projects = [
    "datascraping",
    "loanapproval",
    "outliers",
    "capd",
    "facedetection",
    "handdetection",
    "diapred",
    "aucana",
    "handwriting",
    "score",
    "lazy",
    "objdet",
    "wabra",
    "fitness",
    "chat",
    "weather",
    "cctv",
    "gun",
    "poRt",
    "urine",
    "course",
    "fraud",
    "stock",
    "pixel",
    "chatgpt",
]
images_projects = [
    Image.open(
        f"images/{name}.{'jpg' if name not in ('map', 'gephi', 'health') else 'png'}"
    )
    for name in image_names_projects
]

projects_data = [
    {
        "title": "Data Scraping Project 1",
        "description": "*It is a Data Scraping project of a website that sells products based on Yoga items.*",
        "details": "- All the data of 64 pages is taken.\n- Used Libraries: `Numpy`, `Pandas`, `Matplotlib`, `csv`, `re`, `etc`.",
        "github_link": "https://github.com/Aruj77/Data-Scraping-Pro-1",
        "image": images_projects[0],
        "technologies": ["Python", "Numpy", "Pandas", "Matplotlib", "csv", "re"],
    },
    {
        "title": "Loan Approval",
        "description": "*This project aims to develop a machine learning model that predicts loan approval decisions based on applicant data. By analyzing various features such as income, credit history, loan amount, and more, the model helps financial institutions make informed decisions regarding loan approvals.*",
        "details": "- Features: Data Processing, EDA, Feature Engineering, Model Building, Model Evaluation, Prediction\n- Used Technology: `Python`, `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `Jupyter Notebook`",
        "github_link": "https://github.com/Aruj77/Loan-Approval",
        "image": images_projects[1],
        "technologies": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Matplotlib",
            "Seaborn",
            "Jupyter Notebook",
        ],
    },
    {
        "title": "Remove Outliers in a DataSet",
        "description": "*In this project you will find a function which will help you in cleaning data by removing all the unwanted outliers.*",
        "details": "- Outliers - These are the values that are not needed in the dataset as the are very far away from the range.\n- For finding these outliers we need to find: q1 = 25%tile q3 = 75%tile iqr = q3-q1 lower fence = q1 - 1.5iqr upper fence = q3 + 1.5iqr",
        "github_link": "https://github.com/Aruj77/Remove-Outliers",
        "image": images_projects[2],
        "technologies": ["Python", "Pandas", "NumPy", "Matplotlib"],
    },
    {
        "title": "Car and Pedestrian Detection",
        "description": "*In this project the code is using harcascade model of machine learning to detect cars and pedestrian.*",
        "details": "- Haar cascade is an algorithm that can detect objects in images, irrespective of their scale in image and location.\n- Haar cascade uses the cascading window, and it tries to compute features in every window and classify whether it could be an object. For more details on its working, refer to this link.\n- Used Libraries: `opencv-python`, `numpy`, `haar Cascade`",
        "github_link": "https://github.com/Aruj77/Car-and-Pedestrian-Detection-Machine-Learning",
        "image": images_projects[3],
        "technologies": ["Python", "OpenCV", "numpy"],
    },
    {
        "title": "Face Detection Using Machine Learning",
        "description": "*This project focuses on developing a face detection system using various computer vision and machine learning techniques. The system can detect faces in images and videos, and can be used for applications such as security, photo organization, and more.*",
        "details": "- Features: Image preprocessing, face detection, drawing bounding boxes, real-time detection.\n- The project structure includes directories for data storage (input images and videos), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for face detection and processing, models, and a requirements file listing the required packages.\n- Used Libraries: `OpenCV`, `Dlib`, `NumPy`, `Matplotlib`.",
        "github_link": "https://github.com/Aruj77/Face-Detection",
        "image": images_projects[4],
        "technologies": ["Python", "OpenCV", "Dlib", "NumPy", "Matplotlib"],
    },
    {
        "title": "Hand Detection Using Machine Learning",
        "description": "*This project focuses on developing a hand detection system using various computer vision and machine learning techniques. The system can detect hands in images and videos, and can be used for applications such as gesture recognition, human-computer interaction, and more.*",
        "details": "- The project structure includes directories for data storage (input images and videos), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for hand detection and processing, models, and a requirements file listing the required packages.\n- Features: Image preprocessing, hand detection, drawing bounding boxes, real-time detection.\n- Used Libraries: `OpenCV`, `Dlib`, `NumPy`, `Matplotlib`",
        "github_link": "https://github.com/Aruj77/Hand-Detection-Machine-Learning",
        "image": images_projects[5],
        "technologies": ["Python", "OpenCV", "Dlib", "NumPy", "Matplotlib"],
    },
    {
        "title": "Diabetes Prediction Using Machine Learning",
        "description": "*This project focuses on developing a diabetes prediction system using various machine learning techniques. The system aims to predict the likelihood of diabetes in individuals based on their medical data, helping in early diagnosis and management of the disease.*",
        "details": "- The project structure includes directories for data storage (input datasets), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, training, and evaluating models, models, and a requirements file listing the required packages.\n- Features: Data preprocessing, feature selection, model training, model evaluation, prediction.\n- Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`",
        "github_link": "https://github.com/Aruj77/Diabeties-Prediction-Machine-Learning",
        "image": images_projects[6],
        "technologies": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Matplotlib",
            "Seaborn",
        ],
    },
    {
        "title": "Auction Analysis Using Machine Learning",
        "description": "*This project focuses on analyzing auction data using various data analysis and visualization techniques. The goal is to uncover patterns, trends, and insights that can inform decision-making processes related to auctions.*",
        "details": "- The project structure includes directories for data storage (input datasets), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing and analyzing data, models, and a requirements file listing the required packages.\n- Features: Data preprocessing, exploratory data analysis, data visualization, statistical analysis, trend analysis.\n- Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `Jupyter NoteBook`",
        "github_link": "https://github.com/Aruj77/Auction-Analysis",
        "image": images_projects[7],
        "technologies": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Matplotlib",
            "Seaborn",
            "Jupyter Notebook",
        ],
    },
    {
        "title": "Handwriting Detection Using Machine Learning",
        "description": "*This project focuses on developing a handwriting detection system using various computer vision and machine learning techniques. The system aims to recognize and interpret handwritten text from images, which can be used for applications such as document digitization, note-taking, and more.*",
        "details": "- The project structure includes directories for data storage (input images), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, detecting, and recognizing handwriting, models, and a requirements file listing the required packages.\n- Features: Image preprocessing, text detection, character recognition, text extraction, real-time detection.\n- Used Libraries: `OpenCV`, `TensorFlow` or `PyTorch`, `NumPy`, `Matplotlib`, `Tesseract OCR`, `Pillow`",
        "github_link": "https://github.com/Aruj77/Handwriting-Detection",
        "image": images_projects[8],
        "technologies": [
            "Python",
            "OpenCV",
            "TensorFlow",
            "PyTorch",
            "NumPy",
            "Matplotlib",
            "Tesseract OCR",
            "Pillow",
        ],
    },
    {
        "title": "Function to Find the Score of a Machine Learning Model",
        "description": "*This project provides a function to evaluate the performance of a machine learning model. It includes various metrics such as accuracy, precision, recall, F1 score, and more, to help understand how well the model is performing.*",
        "details": "- The project structure includes directories for data storage (input datasets), Jupyter Notebooks for model training and evaluation, Python scripts for evaluating model performance, models, and a requirements file listing the required packages.\n- Features: Model training, model evaluation, performance metrics calculation, visualization of results.\n- Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`",
        "github_link": "https://github.com/Aruj77/Function-to-Find-the-Score-of-a-Machine-Learning-Model",
        "image": images_projects[9],
        "technologies": ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib"],
    },
    {
        "title": "All Machine Learning Model Score Prediction",
        "description": "*This project leverages LazyPredict to quickly evaluate and compare the performance of multiple machine learning models on a given dataset. LazyPredict provides a simple way to run a variety of models and get their scores without extensive coding.*",
        "details": "- The project structure includes directories for data storage (input datasets), Jupyter Notebooks for running LazyPredict, Python scripts for evaluating model performance, models, and a requirements file listing the required packages.\n- Features: Model training, model evaluation, performance metrics calculation, comparison of multiple models.\n- Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `LazyPredict`",
        "github_link": "https://github.com/Aruj77/All-ML-Model-Score-Prediction",
        "image": images_projects[10],
        "technologies": ["Pandas", "NumPy", "Scikit-learn", "LazyPredict"],
    },
    {
        "title": "Object Detection",
        "description": "*This project focuses on developing an object detection system using the YOLO (You Only Look Once) algorithm. The system aims to detect and classify objects in images and videos in real-time, which can be used for applications such as surveillance, autonomous driving, and more.*",
        "details": "- The project structure includes directories for data storage (input images and videos), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for object detection and processing, models, and a requirements file listing the required packages.\n- Features: Real-time object detection, object localization, image preprocessing, model training, model evaluation.\n- Used Libraries: `OpenCV`, `YOLOv3`, `NumPy`, `Matplotlib`",
        "github_link": "https://github.com/Aruj77/Object-Detection-Machine-Learning",
        "image": images_projects[11],
        "technologies": ["OpenCV", "YOLOv3", "NumPy", "Matplotlib"],
    },
    {
        "title": "Wabra Chat Analyzer",
        "description": "This project focuses on analyzing chat data to extract sentiments and provide analytical insights. It aims to identify sentiment (positive, negative, neutral) and generate reports for customer service improvement.",
        "details": """
            - The project structure includes directories for data storage (input chat logs), Jupyter Notebooks for analysis, Python scripts for preprocessing and analysis, and a requirements file.
            - Features: Sentiment analysis, analytical reporting, data visualization.
            - Used Libraries: `Pandas`, `NLTK`, `Scikit-learn`, `Matplotlib`, `Seaborn`
            """,
        "github_link": "https://github.com/Aruj77/Wabra-Chat-Analyzer",
        "image": images_projects[12],
        "technologies": ["Pandas", "NLTK", "Scikit-learn", "Matplotlib", "Seaborn"],
    },
    {
        "title": "Fitness Club",
        "description": "*This project focuses on developing a responsive and interactive gym website using React. The website aims to provide information about the gym, its services, trainers, membership plans, and more, while also allowing users to book classes and contact the gym.*",
        "details": """
            - The project structure includes directories for components (React components), pages (different pages of the website), services (API calls and services), assets (images, icons, styles), and a configuration file listing the required packages.
            - Features: Responsive design, dynamic content rendering, booking system, contact form, integration with external APIs.
            - Used Libraries: `Node.js`, `npm` or `yarn`, `React`, `React Router`, `Redux`, `Axios`, Styled Components or `CSS/SASS`, `Formik` for forms, `Yup` for form validation.
            """,
        "github_link": "https://github.com/Aruj77/Fitness-Club-",
        "live_demo_link": "https://fitnessone.netlify.app/",
        "image": images_projects[13],
        "technologies": [
            "Node.js",
            "React",
            "Redux",
            "Axios",
            "Styled Components",
            "Formik",
            "Yup",
        ],
    },
    {
        "title": "Chat App",
        "description": "*This project focuses on developing a real-time chat application using Flutter. The app aims to provide a seamless and interactive messaging experience with features such as user authentication, real-time messaging, media sharing, and more.*",
        "details": """
            - The project structure includes directories for UI components (widgets), screens (different pages of the app), services (Firebase services), models (data models), and a configuration file listing the required packages.
            - Features: User authentication, real-time messaging, media sharing (images, videos), push notifications, user profiles, chat rooms/groups.
            - Used Libraries: `Flutter`, `Dart`, `Firebase` (Firestore, Authentication, Storage), `Provider` or `Bloc` for state management.
            """,
        "github_link": "https://github.com/Aruj77/Chat-App",
        "image": images_projects[14],
        "technologies": ["Flutter", "Dart", "Firebase", "Provider", "Bloc"],
    },
    {
        "title": "Weather App",
        "description": "*This project focuses on developing a weather application using React. The app aims to provide real-time weather information, forecasts, and weather-related alerts for various locations.*",
        "details": """
            - The project structure includes directories for components (React components), pages (different pages of the application), services (API calls), assets (images, icons, styles), and a configuration file listing the required packages.
            - Features: Real-time weather updates, weather forecasts, location-based weather, search functionality, weather alerts, responsive design.
            - Used Libraries: `Node.js`, `npm` or `yarn`, `React`, `Axios` (for API calls), `React Router` (for navigation), Styled Components or `CSS/SASS`, `OpenWeatherMap API`, `Tailwind`.
            """,
        "github_link": "https://github.com/Aruj77/weather-app-react",
        "live_demo_link": "https://weatherapp-4d642d49z-aruj77.vercel.app/",
        "image": images_projects[15],
        "technologies": [
            "Node.js",
            "React",
            "Axios",
            "React Router",
            "Styled Components",
            "OpenWeatherMap API",
            "Tailwind",
        ],
    },
    {
        "title": "Reidentification in CCTV",
        "description": "*This project focuses on developing a system for reidentifying individuals in CCTV footage using the YOLO (You Only Look Once) object detection algorithm. The goal is to accurately detect and track individuals across multiple camera feeds for applications such as security, surveillance, and forensic analysis.*",
        "details": """
            - The project structure includes directories for data storage (input videos and images), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, detecting, and reidentifying individuals, models, and a requirements file listing the required packages.
            - Features: Object detection, reidentification, multi-camera tracking, real-time processing, data visualization.
            - Used Libraries: `OpenCV`, `TensorFlow` or `PyTorch`, `NumPy`, `Matplotlib`, `YOLOv3` or `YOLOv4` weights and configuration files, `Scikit-learn`.
            """,
        "github_link": "https://github.com/Aruj77/Reidentification-CCTV-footage-using-yolo",
        "image": images_projects[16],
        "technologies": [
            "OpenCV",
            "TensorFlow",
            "PyTorch",
            "NumPy",
            "YOLOv3",
            "YOLOv4",
            "Scikit-learn",
        ],
    },
    {
        "title": "Person, Gun, and Number Plate Detection",
        "description": "*This project focuses on developing a detection system using the YOLO (You Only Look Once) algorithm to identify persons, guns, and number plates in images and videos. The goal is to provide a robust solution for security and surveillance applications.*",
        "details": """
            - The project structure includes directories for data storage (input videos and images), Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, detecting, and analyzing data, models, and a requirements file listing the required packages.
            - Features: Object detection for persons, guns, and number plates, real-time processing, bounding boxes, alert generation.
            - Used Libraries: `OpenCV`, `TensorFlow` or `PyTorch`, `NumPy`, `Matplotlib`, `YOLOv3` or `YOLOv4` weights and configuration files, `Scikit-learn`.
            """,
        "github_link": "https://github.com/Aruj77/Yolo-model-person-gun-and-number-plate-detection-",
        "image": images_projects[17],
        "technologies": [
            "OpenCV",
            "TensorFlow",
            "PyTorch",
            "NumPy",
            "YOLOv3",
            "YOLOv4",
            "Scikit-learn",
        ],
    },
    {
        "title": "poRt",
        "description": "*This project focuses on developing a responsive and user-friendly website for selling second-hand laptops. The website provides features for listing laptops, searching for laptops, user authentication, and secure transactions.*",
        "details": """
            - The project structure includes separate directories for frontend and backend code, each containing components, pages, services, models, routes, and configuration files.
            - Features: User authentication, laptop listings, search functionality, filtering options, detailed laptop descriptions, user profiles, secure transactions, responsive design.
            - Used Libraries: `Node.js`, `npm` or `yarn`, `React`, `Redux` (for state management), `Axios` (for API calls), `React Router` (for navigation), Styled Components or `CSS/SASS`, `Express` (for backend), `MongoDB` (for database).
            """,
        "github_link": "https://github.com/Aruj77/poRt",
        "image": images_projects[18],
        "technologies": [
            "Node.js",
            "React",
            "Redux",
            "Axios",
            "React Router",
            "Styled Components",
            "Express",
            "MongoDB",
        ],
    },
    {
        "title": "Urine Strip Analyzer",
        "description": "*This project focuses on developing a Urine Strip Analyzer application that uses image processing to analyze urine strips. The backend is developed using Django, the frontend uses HTML, CSS, and JavaScript (jQuery), and image processing is done using OpenCV. The database used is SQLite.*",
        "details": """
            - The project structure includes directories for the backend (Django), frontend (HTML/CSS/JavaScript), image processing (OpenCV scripts), and database (SQLite).
            - Features: User authentication, urine strip image upload, image processing for analysis, results display, user profile management, historical data storage.
            - Used Libraries: `Django`, `OpenCV`, `SQLite`, `HTML`, `CSS`, and `JavaScript` (jQuery).
            """,
        "github_link": "https://github.com/Aruj77/Urine-Strip-Analyzer",
        "live_demo_link": "https://urine-strip-analyzer.vercel.app/",
        "image": images_projects[19],
        "technologies": [
            "Django",
            "HTML",
            "CSS",
            "JavaScript",
            "jQuery",
            "OpenCV",
            "SQLite",
        ],
    },
    {
        "title": "Course List and Details",
        "description": "*This project focuses on developing a responsive and user-friendly website for listing courses and displaying their details. The frontend is developed using React, and the website provides features such as course listings, search functionality, filtering options, and detailed course descriptions.*",
        "details": """
            - The project structure includes directories for components (React components), pages (different pages of the application), services (API calls), assets (images, icons, styles), and a configuration file listing the required packages.
            - Features: User authentication, course listings, search functionality, filtering options, detailed course descriptions, user profiles, responsive design.
            - Used Libraries: `Node.js`, `npm` or `yarn`, `React`, `Redux` (for state management), `Axios` (for API calls), `React Router` (for navigation), Styled Components or `CSS/SASS`.
            """,
        "github_link": "https://github.com/Aruj77/Course-List-and-details-React-",
        "live_demo_link": "https://course-list-and-details-react.vercel.app/",
        "image": images_projects[20],
        "technologies": [
            "Node.js",
            "React",
            "Redux",
            "Axios",
            "React Router",
            "Styled Components",
        ],
    },
    {
        "title": "Fraudulent Data Analysis",
        "description": "*This project focuses on analyzing data to detect fraudulent activities using various data science and machine learning techniques. The aim is to develop a system that can identify anomalies and patterns indicative of fraud.*",
        "details": """
            - The project structure includes directories for data storage, Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, feature engineering, model training, and evaluation, and a requirements file listing the required packages.
            - Features: Data preprocessing, feature engineering, exploratory data analysis, model training and evaluation, anomaly detection, visualization of results.
            - Used Libraries: `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `TensorFlow` or `PyTorch`, `Jupyter Notebook`.
            """,
        "github_link": "https://github.com/Aruj77/Fraudulent-Data-Analysis",
        "image": images_projects[21],
        "technologies": [
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Matplotlib",
            "Seaborn",
            "TensorFlow",
            "PyTorch",
            "Jupyter Notebook",
        ],
    },
    {
        "title": "Stock Monitoring Platform",
        "description": "*This project focuses on developing a Stock Monitoring Platform that allows users to track and analyze stock market data. The platform provides real-time stock updates, historical data analysis, visualizations, and alerts for significant market events.*",
        "details": """
            - The project structure includes directories for data storage, Jupyter Notebooks for exploratory analysis and experimentation, Python scripts for preprocessing, feature engineering, model training, and evaluation, and a requirements file listing the required packages.
            - Features: Real-time stock data updates, historical data analysis, stock performance visualization, user-defined alerts, portfolio tracking, user authentication.
            - Used Libraries: `Django`, `Pandas`, `NumPy`, `Matplotlib`, `Plotly`, `Requests`, `SQLite` (default for Django), `React.js`, `CSS`, `JavaScript` (jQuery)
            """,
        "github_link": "https://github.com/Aruj77/Stock-Monitoring-Platform",
        "image": images_projects[22],
        "technologies": [
            "Django",
            "Pandas",
            "NumPy",
            "Matplotlib",
            "Plotly",
            "Requests",
            "SQLite",
            "React.js",
            "CSS",
            "JavaScript",
            "jQuery",
        ],
    },
    {
        "title": "PixelCops: Image Authentication and Origin Tracing",
        "description": "*In today's digital age, distinguishing authentic images from those generated by AI, especially in sensitive areas like news and social media, is increasingly challenging. PixelCops addresses this crucial need by offering a reliable solution for verifying and tracing the origin of digital content. This system analyzes images pixel by pixel to detect manipulations and identify the source of the images.*",
        "details": """
            - The project structure includes directories for the backend (Django), frontend (HTML/CSS/JavaScript), image analysis scripts (OpenCV and machine learning models), and database (SQLite).
            - Features: Pixel-by-pixel image analysis, AI-generated image detection, image manipulation detection, origin tracing, user authentication, detailed reports, responsive design.
            - Used Libraries: `Solidity`, `Rust`, `React Native`, `TypeScript`, `React.js`, `RollUp`, `RISC0`, `ZKVM`
            """,
        "github_link": "https://github.com/Aruj77/PixelCops",
        "image": images_projects[23],
        "technologies": [
            "Solidity",
            "Rust",
            "React Native",
            "TypeScript",
            "React.js",
            "RollUp",
            "RISC0",
            "ZKVM",
        ],
    },
    {
        "title": "GPT Clone",
        "description": "*This project aims to create a conversational AI system similar to OpenAI's ChatGPT. The system will use natural language processing techniques to understand and generate human-like responses in text-based conversations.*",
        "details": """
            - Features: Natural language understanding (NLU) for processing user inputs, Response generation using machine learning models, Context management to maintain coherent conversations, User authentication and personalized responses, Integration with messaging platforms
            - Used Libraries: `PyTorch`, `Transformers library` (for pre-trained language models), Flask (for API endpoints), `React.js`, `CSS`, `JavaScript` (for frontend if applicable), `SQLite` or `MongoDB` (for storing conversation history and context).
            """,
        "github_link": "https://github.com/Aruj77/GPT-Clone",
        "image": images_projects[24],
        "technologies": [
            "PyTorch",
            "Transformers",
            "Flask",
            "React.js",
            "CSS",
            "JavaScript",
            "SQLite",
            "MongoDB",
        ],
    },
]
