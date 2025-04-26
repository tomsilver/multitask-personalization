/***********************************************************
 * CONFIGURATION
 ***********************************************************/

// Question metadata.
const QUESTIONS = [
  {
    key: "feeding_side",
    text: "Which side should the robot feed you from?",
    contentDir: "feeding_side"
  },
  {
    key: "bite_order",
    text: "What bite order would you like?",
    contentDir: "bite_ordering"
  },
  {
    key: "ready_signal",
    text: "What ready signal would you like the robot to use?",
    contentDir: "ready_signal"
  },
  {
    key: "verbal",
    text: "Would you like the robot to be verbal?",
    contentDir: "be_verbal"
  },
  {
    key: "look_forward",
    text: "Would you typically be looking forward during this meal?",
    contentDir: "occlusion"
  },
  {
    key: "block_forward",
    text: "Is the robot uncomfortably blocking your forward sight?",
    contentDir: "occlusion"
  },
  {
    key: "look_left",
    text: "Would you typically be looking a little towards the left during this meal?",
    contentDir: "occlusion"
  },
  {
    key: "block_left",
    text: "Is the robot uncomfortably blocking your leftward sight?",
    contentDir: "occlusion"
  },
];

// Fallback option lists for the very first meal.
// These will be replaced automatically per-meal using prediction files.
const INITIAL_OPTIONS = {
  feeding_side: ["Loading..."],
  bite_order: ["Loading..."],
  ready_signal: ["Loading..."],
  verbal: ["Loading..."],
  look_forward: ["Yes", "No"],
  block_forward: ["Yes", "No"],
  look_left: ["Yes", "No"],
  block_left: ["Yes", "No"],
};

/***********************************************************
 * STATE
 ***********************************************************/
const state = {
  answers: [], // Array of per‐meal answer objects
  metadata: {}, // Stores loaded metadata per question
  predictions: {}, // Stores loaded predictions per question
  currentMeal: null, // Stores the current meal metadata
};

/***********************************************************
 * DOM ELEMENTS
 ***********************************************************/
const homeScreen = document.getElementById("home-screen");
const mealScreen = document.getElementById("meal-screen");
const thanksScreen = document.getElementById("thanks-screen");
const startBtn = document.getElementById("start-btn");
const nextBtn = document.getElementById("next-btn");
const mealTitle = document.getElementById("meal-title");
const mealImg = document.getElementById("meal-img");
const mealDesc = document.getElementById("meal-desc");
const formEl = document.getElementById("questions-form");

/***********************************************************
 * UTILS
 ***********************************************************/
function show(el) {
  el.classList.remove("hidden");
}

function hide(el) {
  el.classList.add("hidden");
}

/**
 * Generates a test URL for a specific series of answers
 * @param {Array} answers - Array of answer objects
 * @returns {string} The URL with encoded answers
 */
function generateTestUrl(answers) {
  const encodedAnswers = encodeURIComponent(JSON.stringify(answers));
  return `meal.html?answers=${encodedAnswers}`;
}

/**
 * Builds the content directory path for a given question based on previous answers.
 * Example: If user chose "left" for feeding_side, then "right" for bite_order,
 * the path for bite_order would be: "content/bite_ordering/left/"
 * 
 * @param {string} questionKey - The key of the current question (e.g., "feeding_side", "bite_order")
 * @returns {string} The full path to the content directory for this question
 */
function getContentPath(questionKey) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) return null;

  // Start with the base content directory and question type
  const pathParts = ['content', question.contentDir];

  // Add previous answers to build the path
  // For example, if we're on bite_order and user chose "left" for feeding_side,
  // we need to include that in the path
  for (const answer of state.answers) {
    // Only include answers for questions that come before the current question
    const questionIndex = QUESTIONS.findIndex(q => q.key === questionKey);
    const answerQuestionIndex = QUESTIONS.findIndex(q => q.key === Object.keys(answer)[0]);
    
    if (answerQuestionIndex < questionIndex) {
      pathParts.push(answer[Object.keys(answer)[0]].value);
    }
  }

  return pathParts.join('/');
}

/***********************************************************
 * STATE MANAGEMENT
 ***********************************************************/
function getStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const answers = params.get('answers') ? JSON.parse(decodeURIComponent(params.get('answers'))) : [];
  return { answers };
}

function updateUrlState() {
  const params = new URLSearchParams();
  params.set('answers', encodeURIComponent(JSON.stringify(state.answers)));
  
  // Update URL without reloading the page
  window.history.replaceState(
    {},
    '',
    `${window.location.pathname}?${params.toString()}`
  );
}

function navigateToMeal() {
  const params = new URLSearchParams();
  if (state.answers.length > 0) {
    params.set('answers', encodeURIComponent(JSON.stringify(state.answers)));
  }
  window.location.href = `meal.html?${params.toString()}`;
}

function navigateToThanks() {
  window.location.href = 'thanks.html';
}

/***********************************************************
 * CONTENT LOADING
 ***********************************************************/
async function loadMetadata(questionKey) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) return null;
  
  try {
    const contentPath = getContentPath(questionKey);
    const response = await fetch(`${contentPath}/metadata.json`);
    return await response.json();
  } catch (error) {
    console.error(`Error loading metadata for ${questionKey}:`, error);
    return null;
  }
}

async function loadPredictions(questionKey) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) return null;
  
  try {
    const contentPath = getContentPath(questionKey);
    const response = await fetch(`${contentPath}/prediction.txt`);
    const text = await response.text();
    return text.trim().split('\n');
  } catch (error) {
    console.error(`Error loading predictions for ${questionKey}:`, error);
    return null;
  }
}

async function loadCurrentMealInfo() {
  // Load meal info from occlusion metadata
  const occlusionPath = getContentPath('look_forward');
  try {
    const response = await fetch(`${occlusionPath}/metadata.json`);
    const metadata = await response.json();
    state.currentMeal = {
      title: `Meal ${state.answers.length + 1} of 5`,
      image: metadata.image || `media/meal_${state.answers.length + 1}.jpg`,
      description: metadata.description || "Please answer the following questions about this meal scenario."
    };
  } catch (error) {
    console.error('Error loading meal info:', error);
    state.currentMeal = {
      title: `Meal ${state.answers.length + 1} of 5`,
      image: `media/meal_${state.answers.length + 1}.jpg`,
      description: "Please answer the following questions about this meal scenario."
    };
  }
}

/***********************************************************
 * UI RENDERING
 ***********************************************************/
async function loadCurrentContent() {
  state.metadata = {};
  state.predictions = {};
  
  // Load meal info first
  await loadCurrentMealInfo();
  
  // Load metadata and predictions for each question
  for (const question of QUESTIONS) {
    const key = question.key;
    state.metadata[key] = await loadMetadata(key);
    state.predictions[key] = await loadPredictions(key);
  }
}

function getOptionsForQuestion(questionKey) {
  // For bite ordering, use the choices from metadata
  if (questionKey === 'bite_order') {
    const metadata = state.metadata[questionKey];
    if (metadata && metadata.choices) {
      return metadata.choices;
    }
  }
  
  // For verbal, always use Yes/No for display
  if (questionKey === 'verbal') {
    return ['Yes', 'No'];
  }
  
  // For other questions, use metadata choices if available, otherwise fall back to initial options
  const metadata = state.metadata[questionKey];
  if (metadata && metadata.choices) {
    return metadata.choices;
  }
  
  return INITIAL_OPTIONS[questionKey];
}

function renderForm() {
  const form = document.getElementById("questions-form");
  form.innerHTML = ""; // Clear existing questions
  
  QUESTIONS.forEach((question) => {
    const options = getOptionsForQuestion(question.key);
    
    const wrapper = document.createElement("div");
    wrapper.style.marginBottom = "1.5rem";
    
    const label = document.createElement("label");
    label.setAttribute("for", question.key);
    label.textContent = question.text;
    
    const select = document.createElement("select");
    select.id = question.key;
    select.name = question.key;
    
    // Add empty default option
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Select an option...";
    select.appendChild(defaultOption);
    
    // Add all available options
    if (Array.isArray(options)) {
      options.forEach((optionText) => {
        if (optionText) {  // Only add non-empty options
          const option = document.createElement("option");
          option.value = optionText;
          option.textContent = optionText;
          select.appendChild(option);
        }
      });
    } else {
      console.warn(`Options for ${question.key} is not an array:`, options);
    }
    
    // If we have metadata for this question, add it as a data attribute
    const metadata = state.metadata[question.key];
    if (metadata) {
      select.dataset.metadata = JSON.stringify(metadata);
    }
    
    // Pre-select the predicted option if available
    const prediction = state.predictions[question.key];
    if (prediction && prediction.length > 0) {
      // For verbal, convert True/False to Yes/No
      if (question.key === 'verbal') {
        const predictedValue = prediction[0].trim();
        select.value = predictedValue === 'True' ? 'Yes' : 'No';
      } else {
        select.value = prediction[0];
      }
    }
    
    select.addEventListener("change", checkFormCompletion);
    
    wrapper.appendChild(label);
    wrapper.appendChild(select);
    form.appendChild(wrapper);
  });
}

async function showMeal() {
  // Load content before showing the meal
  await loadCurrentContent();
  
  const mealTitle = document.getElementById("meal-title");
  const mealImg = document.getElementById("meal-img");
  const mealDesc = document.getElementById("meal-desc");
  
  mealTitle.textContent = state.currentMeal.title;
  mealImg.src = state.currentMeal.image;
  mealDesc.textContent = state.currentMeal.description;
  
  renderForm();
}

function checkFormCompletion() {
  const form = document.getElementById("questions-form");
  const nextBtn = document.getElementById("next-btn");
  const selects = form.querySelectorAll("select");
  
  // Check if all selects have a value
  const allAnswered = Array.from(selects).every(select => select.value);
  nextBtn.disabled = !allAnswered;
}

function collectAnswers() {
  const form = document.getElementById("questions-form");
  const answers = {};
  
  QUESTIONS.forEach(question => {
    const select = form.querySelector(`select[name="${question.key}"]`);
    answers[question.key] = {
      value: select.value,
      metadata: select.dataset.metadata ? JSON.parse(select.dataset.metadata) : null
    };
  });
  
  return answers;
}

function finishStudy() {
  // Hide meal screen
  document.getElementById("meal-screen").classList.add("hidden");
  // Show thanks screen
  document.getElementById("thanks-screen").classList.remove("hidden");
  
  // Here you could send the state.answers data to your server
  console.log("Study completed!", state.answers);
}

/***********************************************************
 * EVENT HANDLERS
 ***********************************************************/
function handleNextClick() {
  // Collect and save answers
  const answers = collectAnswers();
  state.answers.push(answers);
  
  if (state.answers.length < 5) {
    navigateToMeal();
  } else {
    navigateToThanks();
  }
}

/***********************************************************
 * INIT
 ***********************************************************/
// Initialize state from URL on page load
const { answers } = getStateFromUrl();
state.answers = answers;

// Set up event listeners based on current page
if (window.location.pathname.endsWith('meal.html')) {
  document.getElementById('next-btn').addEventListener('click', handleNextClick);
  showMeal();
} else if (window.location.pathname.endsWith('index.html') || window.location.pathname.endsWith('/')) {
  // Handle the start button on the index page
  document.getElementById('start-btn').addEventListener('click', () => {
    navigateToMeal();
  });
}

// Add browser back/forward button support
window.addEventListener('popstate', () => {
  const { answers } = getStateFromUrl();
  state.answers = answers;
  if (window.location.pathname.endsWith('meal.html')) {
    showMeal();
  }
}); 