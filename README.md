# 🌤️ AtmoXplorer  
**Meteorological Data Analysis Web App**  
**Internship Project at ITR, DRDO**

---

## 📖 About the Project

**AtmoXplorer** is a lightweight, offline-capable web application developed as part of our internship at **Integrated Test Range (ITR), DRDO**. The tool is designed to process, clean, and visualize atmospheric data collected from weather instruments such as ISRO and Komoline loggers. It helps researchers and engineers easily analyze parameters like temperature, pressure, humidity, and wind direction/speed at various altitudes.

---

## ✨ Key Features

- 📂 Upload CSV data from **ISRO** or **Komoline** weather instruments
- 🧼 Automatic data cleaning (e.g., removes rows with non-positive altitudes)
- 🔁 Wind speed unit conversion (Knots ➝ m/s for Komoline)
- 🧮 Auto-generation of standard altitude levels for smooth visualization
- 📊 Multiple interactive visualizations:
  - Temperature vs Altitude  
  - Pressure vs Altitude  
  - Humidity vs Altitude  
  - Wind Speed vs Altitude  
  - Wind Direction vs Altitude  
  - Wind Rose Plot
- 💻 Built for **offline use** (packaged for Linux & Windows)
- 🌙 Dark/light mode support for better usability

---

## 🛠️ Built With

- **Python** (Flask) – Backend logic & server
- **HTML / CSS / JavaScript** – Frontend UI
- **Chart.js** – Interactive plotting
- **Pandas, NumPy** – Data handling and transformations
- **PyInstaller** – App packaging (Linux AppImage / Windows EXE)

---

## 📁 Folder Structure

```
AtmoXplorer/
├── static/                 # CSS and JavaScript files
├── templates/              # HTML templates (Jinja2)
├── Ubuntu/                 # Files for Ubuntu AppImage build
│   ├── AtmoXplorer.desktop
│   ├── README.txt
│   └── requirement.txt
├── app.py / main.py        # Flask app entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

---

## 🚀 Getting Started (Run Locally)

### 🔹 Prerequisites
- Python 3.8+
- pip

### 🔹 Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ronak-001/AtmoXplorer-Internship-project-.git
   cd AtmoXplorer-Internship-project-
   ```

2. **Create a virtual environment (optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python app.py
   ```
   Open your browser and visit: `http://127.0.0.1:5000`

---

## 📦 Offline Linux Executable (AppImage)

1. Navigate to the `Ubuntu/` directory.
2. Use the provided `AtmoXplorer.AppImage` to run the app on any Ubuntu-based system.
3. Optional `.desktop` launcher and `README.txt` included for user convenience.

---

## 📷 Screenshots


![Upload Page](images/Screenshot%202025-07-09%20192312.png)

![CSV Preview](images/Screenshot%202025-07-09%20192342.png)

![Graphs View](images/Screenshot%202025-07-09%20192358.png)

![Wind Rose](images/Screenshot%202025-07-09%20192413.png)

![Auto Generation](images/Screenshot%202025-07-09%20193210.png)

![Final UI](images/Screenshot%202025-07-09%20194140.png)

![Final UI](images/Screenshot%2025-07-24%101449.png)

