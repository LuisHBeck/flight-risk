# ✈️ Flight Risk

> Machine Learning project for predicting delays in Brazilian domestic flights using historical ANAC data.

## 📖 Overview

Flight Risk is a Machine Learning project designed to predict the probability of delays in domestic flights in Brazil based on official historical data provided by ANAC (Brazilian National Civil Aviation Agency).

The system aims to help passengers and businesses anticipate potential delays, enabling better travel planning and operational decision-making.

---

## 🎯 Objectives

* Predict the probability of flight delays.
* Analyze historical flight performance data.
* Identify the main factors associated with delays.
* Provide actionable insights for travelers and aviation stakeholders.

---

<details>
<summary><strong>🧑‍💻 Development Setup</strong></summary>

### 1. Clone the Repository

```sh
git clone --branch develop git@github.com:LuisHBeck/flight-risk.git
cd flight-risk
```

### 2. Create Your Feature Branch

Replace `<ghUser>` with your GitHub username.

```sh
git checkout -b <ghUser> develop
```

### 3. Create the Python Environment

```sh
pyenv virtualenv 3.12.9 flight-risk
pyenv local flight-risk
```

### 4. Run the Initial Setup

Install project dependencies and generate the `.env` file.

```sh
make dev_setup
```

### 5. Configure Environment Variables

Open the generated `.env` file and fill in the required values.
```sh
direnv reload .
```

### 6. Download the Consolidated Dataset

```sh
make fetch_consolidated_dataset
```

</details>

---
