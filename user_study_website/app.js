/***********************************************************
 * CONFIGURATION
 ***********************************************************/
const TOTAL_MEALS = 5; // N rounds

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
  verbal: ["Yes", "No"],
  look_forward: ["Yes", "No"],
  block_forward: ["Yes", "No"],
  look_left: ["Yes", "No"],
  block_left: ["Yes", "No"],
};

// Meal scenarios with their metadata
const MEALS = [
    {
      title: "Meal 1 of 5",
      scenario: "french fries dipped in ketchup",
      image: "media/meal_1.jpg",
    },
    {
      title: "Meal 2 of 5",
      scenario: "french fries without any dipping",
      image: "media/meal_2.jpg",
    },
    {
      title: "Meal 3 of 5",
      scenario: "french fries dipped in ranch dressing",
      image: "media/meal_3.jpg",
    },
    // Add more meals as needed
];

/***********************************************************
 * CONTENT LOADING
 ***********************************************************/
async function loadMetadata(questionKey, scenario) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) return null;
  
  try {
    const response = await fetch(`content/${question.contentDir}/${scenario}/metadata.json`);
    return await response.json();
  } catch (error) {
    console.error(`Error loading metadata for ${questionKey}/${scenario}:`, error);
    return null;
  }
}

async function loadPredictions(questionKey, scenario) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) return null;
  
  try {
    const response = await fetch(`content/${question.contentDir}/${scenario}/prediction.txt`);
    const text = await response.text();
    return text.trim().split('\n');
  } catch (error) {
    console.error(`Error loading predictions for ${questionKey}/${scenario}:`, error);
    return null;
  }
}

/***********************************************************
 * STATE
 ***********************************************************/
const state = {
  round: 0,
  answers: [], // Array of per‐meal answer objects
  metadata: {}, // Stores loaded metadata per question
  predictions: {}, // Stores loaded predictions per question
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

/***********************************************************
 * STATE MANAGEMENT
 ***********************************************************/
function getStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const round = parseInt(params.get('round') || '0');
  const answers = params.get('answers') ? JSON.parse(decodeURIComponent(params.get('answers'))) : [];
  return { round, answers };
}

function updateUrlState() {
  const params = new URLSearchParams();
  params.set('round', state.round.toString());
  params.set('answers', encodeURIComponent(JSON.stringify(state.answers)));
  
  // Update URL without reloading the page
  window.history.replaceState(
    {},
    '',
    `${window.location.pathname}?${params.toString()}`
  );
}

function navigateToMeal(round) {
  const params = new URLSearchParams();
  params.set('round', round.toString());
  if (state.answers.length > 0) {
    params.set('answers', encodeURIComponent(JSON.stringify(state.answers)));
  }
  window.location.href = `meal.html?${params.toString()}`;
}

function navigateToThanks() {
  window.location.href = 'thanks.html';
}

function restoreState() {
  const { round, answers } = getStateFromUrl();
  state.round = round;
  state.answers = answers;
  
  // If we have answers, we're in the middle of the study
  if (answers.length > 0) {
    hide(homeScreen);
    show(mealScreen);
    showMeal();
  }
}

/***********************************************************
 * UI RENDERING
 ***********************************************************/
async function loadCurrentMealContent() {
  const meal = MEALS[state.round];
  state.metadata = {};
  state.predictions = {};
  
  // Load metadata and predictions for each question
  for (const question of QUESTIONS) {
    state.metadata[question.key] = await loadMetadata(question.key, meal.scenario);
    state.predictions[question.key] = await loadPredictions(question.key, meal.scenario);
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
  
  // For other questions, use predictions if available, otherwise fall back to initial options
  return state.predictions[questionKey] || INITIAL_OPTIONS[questionKey];
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
    
    // Add predicted/available options
    options.forEach((optionText) => {
      const option = document.createElement("option");
      option.value = optionText;
      option.textContent = optionText;
      select.appendChild(option);
    });
    
    // If we have metadata for this question, add it as a data attribute
    const metadata = state.metadata[question.key];
    if (metadata) {
      select.dataset.metadata = JSON.stringify(metadata);
    }
    
    select.addEventListener("change", checkFormCompletion);
    
    wrapper.appendChild(label);
    wrapper.appendChild(select);
    form.appendChild(wrapper);
  });
}

async function showMeal() {
  const meal = MEALS[state.round];
  
  // Load content before showing the meal
  await loadCurrentMealContent();
  
  const mealTitle = document.getElementById("meal-title");
  const mealImg = document.getElementById("meal-img");
  const mealDesc = document.getElementById("meal-desc");
  
  mealTitle.textContent = meal.title;
  mealImg.src = meal.image;
  
  // Use metadata for description if available
  const description = state.metadata?.description || meal.scenario;
  mealDesc.textContent = description;
  
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
document.getElementById("start-btn").addEventListener("click", () => {
  document.getElementById("home-screen").classList.add("hidden");
  document.getElementById("meal-screen").classList.remove("hidden");
  state.round = 0;
  state.answers = [];
  updateUrlState();
  showMeal();
});

function handleNextClick() {
  // Collect and save answers
  const answers = collectAnswers();
  state.answers.push(answers);
  
  // Move to next round
  state.round += 1;
  
  if (state.round < MEALS.length) {
    navigateToMeal(state.round);
  } else {
    navigateToThanks();
  }
}

/***********************************************************
 * INIT
 ***********************************************************/
// Initialize state from URL on page load
const { round, answers } = getStateFromUrl();
state.round = round;
state.answers = answers;

// Set up event listeners based on current page
if (window.location.pathname.includes('meal.html')) {
  document.getElementById('next-btn').addEventListener('click', handleNextClick);
  showMeal();
}

// Add browser back/forward button support
window.addEventListener('popstate', () => {
  restoreState();
}); 