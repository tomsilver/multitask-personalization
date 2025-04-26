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
    text: "Would you like the robot to be verbal during this meal?",
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
    // AND are relevant to the current question's content directory
    const questionIndex = QUESTIONS.findIndex(q => q.key === questionKey);
    const answerQuestionIndex = QUESTIONS.findIndex(q => q.key === Object.keys(answer)[0]);
    
    if (answerQuestionIndex < questionIndex) {
      const answerQuestion = QUESTIONS[answerQuestionIndex];
      // Only include the answer if it's from the same content directory
      if (answerQuestion.contentDir === question.contentDir) {
        pathParts.push(answer[Object.keys(answer)[0]].value);
      }
    }
  }

  // Special handling for occlusion directory naming
  if (question.contentDir === 'occlusion') {
    // Get the last answer to determine the occlusion state
    const lastAnswer = state.answers[state.answers.length - 1];
    if (lastAnswer) {
      const relevantPois = [];
      const occludedPois = [];
      
      // Check each occlusion-related answer
      if (lastAnswer.look_forward?.value === 'Yes') relevantPois.push('front');
      if (lastAnswer.look_left?.value === 'Yes') relevantPois.push('left');
      if (lastAnswer.block_forward?.value === 'Yes') occludedPois.push('front');
      if (lastAnswer.block_left?.value === 'Yes') occludedPois.push('left');
      
      // Create the directory name
      const relevantStr = relevantPois.length > 0 ? relevantPois.join('-') : 'none';
      const occludedStr = occludedPois.length > 0 ? occludedPois.join('-') : 'none';
      const occlusionDir = `${relevantStr}___${occludedStr}`;
      
      // Append the occlusion directory instead of replacing
      pathParts.push(occlusionDir);
    }
  }

  return pathParts.join('/');
}

/***********************************************************
 * STATE MANAGEMENT
 ***********************************************************/
function compressState(answers) {
  // Convert answers to a minimal format
  const minimalAnswers = answers.map(answer => {
    const minimal = {};
    for (const [key, value] of Object.entries(answer)) {
      if (key === 'occlusion') {
        // Special handling for occlusion data
        minimal[key] = {
          r: value.relevant_pois,
          o: value.occluded_pois
        };
      } else {
        // For other answers, just store the value
        minimal[key] = value.value;
      }
    }
    return minimal;
  });
  
  // Convert to base64
  const json = JSON.stringify(minimalAnswers);
  return btoa(json);
}

function decompressState(compressed) {
  try {
    // Decode base64
    const json = atob(compressed);
    const minimalAnswers = JSON.parse(json);
    
    // Convert back to full format
    return minimalAnswers.map(answer => {
      const full = {};
      for (const [key, value] of Object.entries(answer)) {
        if (key === 'occlusion') {
          // Special handling for occlusion data
          full[key] = {
            relevant_pois: value.r,
            occluded_pois: value.o
          };
        } else {
          // For other answers, restore the full format
          full[key] = {
            value: value,
            metadata: null
          };
        }
      }
      return full;
    });
  } catch (error) {
    console.error('Error decompressing state:', error);
    return [];
  }
}

function getStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const compressed = params.get('state');
  const answers = compressed ? decompressState(compressed) : [];
  return { answers };
}

function updateUrlState() {
  const params = new URLSearchParams();
  const compressed = compressState(state.answers);
  params.set('state', compressed);
  
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
    const compressed = compressState(state.answers);
    params.set('state', compressed);
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
    console.log(`Loading metadata from: ${contentPath}/metadata.json`);
    
    const response = await fetch(`${contentPath}/metadata.json`);
    if (!response.ok) {
      console.error(`HTTP error loading metadata for ${questionKey}: ${response.status} ${response.statusText}`);
      return null;
    }
    
    const text = await response.text();
    if (!text.trim()) {
      console.error(`Empty metadata file for ${questionKey}`);
      return null;
    }
    
    try {
      return JSON.parse(text);
    } catch (parseError) {
      console.error(`Invalid JSON in metadata for ${questionKey}:`, text);
      return null;
    }
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
    if (questionKey.startsWith('look_') || questionKey.startsWith('block_')) {
      // For occlusion questions, load prediction.json instead of prediction.txt
      const response = await fetch(`${contentPath}/prediction.json`);
      const data = await response.json();
      return data;
    } else {
      const response = await fetch(`${contentPath}/prediction.txt`);
      const text = await response.text();
      return text.trim().split('\n');
    }
  } catch (error) {
    console.error(`Error loading predictions for ${questionKey}:`, error);
    return null;
  }
}

async function loadCurrentMealInfo() {
  // Load meal info from occlusion metadata
  const occlusionPath = getContentPath('look_forward');
  try {
    console.log('Attempting to load meal metadata:', occlusionPath); // Debug log
    const response = await fetch(`${occlusionPath}/metadata.json`);
    const metadata = await response.json();
    console.log('Loaded meal metadata:', metadata); // Debug log
    
    // Create a descriptive meal context
    const foodItems = metadata.food_items.join(' and ');
    const dips = metadata.dips.join(' and ');
    const context = metadata.context.replace('_', ' '); // Convert "personal" to "personal"
    const tableType = metadata.table_type.replace('_', ' '); // Convert "rectangular_table" to "rectangular table"
    
    const description = `Imagine you are having a meal in a <span class="context">${context}</span> setting at a <span class="table-type">${tableType}</span>. 
    On your plate, you have <span class="food-items">${foodItems}</span>${dips ? ` with <span class="dips">${dips}</span> for dipping` : ''}. 
    Please answer the following questions about this meal scenario.`;
    
    state.currentMeal = {
      title: `Meal ${state.answers.length + 1} of 5`,
      image: 'content/occlusion/bite_occlusion_image.png',
      description: description
    };
    
    console.log('Final image path:', state.currentMeal.image); // Debug log
  } catch (error) {
    console.error('Error loading meal info:', error);
    state.currentMeal = {
      title: `Meal ${state.answers.length + 1} of 5`,
      image: 'content/occlusion/bite_occlusion_image.png',
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
  
  // For occlusion questions, always use Yes/No
  if (questionKey.startsWith('look_') || questionKey.startsWith('block_')) {
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
    label.className = 'meal-context';
    
    // Get the prediction for this question
    let prediction;
    if (question.key.startsWith('look_') || question.key.startsWith('block_')) {
      const occlusionData = state.predictions['look_forward'];
      if (occlusionData) {
        const direction = question.key.includes('forward') ? 'front' : 'left';
        if (question.key.startsWith('look_')) {
          prediction = occlusionData.relevant_pois.includes(direction) ? 'Yes' : 'No';
        } else {
          prediction = occlusionData.occluded_pois.includes(direction) ? 'Yes' : 'No';
        }
      }
    } else {
      const pred = state.predictions[question.key];
      if (pred && pred.length > 0) {
        if (question.key === 'verbal') {
          prediction = pred[0].trim() === 'True' ? 'Yes' : 'No';
        } else if (question.key === 'bite_order') {
          const index = parseInt(pred[0].trim());
          if (!isNaN(index) && index >= 0 && index < options.length) {
            prediction = options[index];
          }
        } else {
          prediction = pred[0];
        }
      }
    }
    
    // Create the question text based on the type
    if (question.key === 'feeding_side') {
      label.innerHTML = `The robot is planning to feed you from the <span class="context">${prediction || 'left'}</span> side. Are you happy with this choice or would you like to choose another?`;
    } else if (question.key === 'bite_order') {
      label.innerHTML = `The robot is planning to serve your food as follows: <span class="context">${prediction || 'alternating bites'}</span>. Are you happy with this choice or would you like to choose another?`;
    } else if (question.key === 'ready_signal') {
      label.innerHTML = `The robot is planning to use <span class="context">${prediction || 'a button'}</span> as a ready signal. Are you happy with this choice or would you like to choose another?`;
    } else if (question.key === 'verbal') {
      label.innerHTML = `Would you like the robot to be <span class="context">verbal</span> during this meal?`;
    } else if (question.key.startsWith('look_')) {
      const direction = question.key.includes('forward') ? 'forward' : 'left';
      label.innerHTML = `Would you typically be looking <span class="context">${direction}</span> during this meal?`;
    } else if (question.key.startsWith('block_')) {
      const direction = question.key.includes('forward') ? 'forward' : 'left';
      label.innerHTML = `Is the robot uncomfortably blocking your <span class="context">${direction}</span> sight?`;
    } else {
      label.textContent = question.text;
    }
    
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
    if (prediction) {
      select.value = prediction;
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
  
  // Set the description with proper HTML
  mealDesc.innerHTML = state.currentMeal.description;
  mealDesc.className = 'meal-context';
  
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
    const value = select.value;
    
    // Handle special cases for storing answers
    if (question.key === 'verbal') {
      // Convert Yes/No back to True/False
      answers[question.key] = {
        value: value === 'Yes' ? 'True' : 'False',
        metadata: select.dataset.metadata ? JSON.parse(select.dataset.metadata) : null
      };
    } else if (question.key === 'bite_order') {
      // Store the index of the selected option
      const options = getOptionsForQuestion(question.key);
      const index = options.indexOf(value);
      answers[question.key] = {
        value: index.toString(),
        metadata: select.dataset.metadata ? JSON.parse(select.dataset.metadata) : null
      };
    } else if (question.key.startsWith('look_') || question.key.startsWith('block_')) {
      // For occlusion questions, we need to update the prediction.json structure
      const direction = question.key.includes('forward') ? 'front' : 'left';
      const isLooking = question.key.startsWith('look_');
      
      // Get or create the occlusion data
      if (!answers.occlusion) {
        answers.occlusion = {
          relevant_pois: [],
          occluded_pois: []
        };
      }
      
      // Update the appropriate list based on the answer
      if (value === 'Yes') {
        if (isLooking) {
          answers.occlusion.relevant_pois.push(direction);
        } else {
          answers.occlusion.occluded_pois.push(direction);
        }
      }
      
      // Store the individual answer as well
      answers[question.key] = {
        value,
        metadata: select.dataset.metadata ? JSON.parse(select.dataset.metadata) : null
      };
    } else {
      // For all other questions, store the value as is
      answers[question.key] = {
        value,
        metadata: select.dataset.metadata ? JSON.parse(select.dataset.metadata) : null
      };
    }
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