# 📂 Dataset Instructions

This project requires an image dataset structured in class-based folders.  

Example:

Dataset/
│── Train/
│ ├── Glare/
│ ├── Ok/
│ └── Scratch/
│
│── Validasi/
│ ├── Glare/
│ ├── Ok/
│ └── Scratch/
│
└── Test/
├── Glare/
├── Ok/
└── Scratch/

- Each folder represents one class (e.g., `OK`, `Defect`).  
- Images will be automatically resized to **128x128** during preprocessing.  
- For larger datasets, place them outside GitHub and provide a download link instead.  
