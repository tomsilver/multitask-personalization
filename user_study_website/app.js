/***********************************************************
 * CONFIGURATION
 ***********************************************************/

// Question metadata.
const QUESTIONS = [
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
  optionMappings: [], // Array of booleans tracking isOptionAPersonalized for each meal
  tempPreferenceRating: null, // Temporarily stored preference rating when transitioning between screens
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
  return `meal-preferences.html?answers=${encodedAnswers}`;
}

/**
 * Builds the content directory path for a given question based on previous answers.
 * Example: If user chose "right" for bite_order,
 * the path for bite_order would be: "content/bite_ordering/right/"
 * 
 * @param {string} questionKey - The key of the current question (e.g., "bite_order")
 * @returns {string} The full path to the content directory for this question
 */
function getContentPath(questionKey) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) return null;

  // Start with the base content directory and question type
  const pathParts = ['content', question.contentDir];

  // Add previous answers to build the path
  // For example, if we're on bite_order and the user chose specific options
  // for previous questions, we need to include those in the path
  if (question.contentDir !== 'occlusion') {
    for (let i = 0; i < state.answers.length; i++) {
      const answer = state.answers[i];
      pathParts.push(answer[questionKey].value);
    }
  }

  // Special handling for occlusion directory naming
  if (question.contentDir === 'occlusion') {
    // Build path by considering each answer's occlusion state in sequence
    for (const answer of state.answers) {
      const relevantPois = [];
      const occludedPois = [];
      
      // Check each occlusion-related answer for this specific answer
      if (answer.look_forward?.value === 'Yes') relevantPois.push('front');
      if (answer.look_left?.value === 'Yes') relevantPois.push('left');
      if (answer.block_forward?.value === 'Yes') occludedPois.push('front');
      if (answer.block_left?.value === 'Yes') occludedPois.push('left');
      
      // Create the directory name for this answer
      const relevantStr = relevantPois.length > 0 ? relevantPois.join('-') : 'none';
      const occludedStr = occludedPois.length > 0 ? occludedPois.join('-') : 'none';
      const occlusionDir = `${relevantStr}___${occludedStr}`;
      
      // Append this answer's occlusion directory to the path
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
      } else if (key === 'isOptionAPersonalized') {
        // Special handling for the option flag - keep it as is
        minimal[key] = value;
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
        } else if (key === 'isOptionAPersonalized') {
          // Special handling for the option flag - keep it as a direct property
          full[key] = value;
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
  const tempRating = params.get('temp_rating');
  const answers = compressed ? decompressState(compressed) : [];
  return { 
    answers,
    tempPreferenceRating: tempRating || null
  };
}

function updateUrlState() {
  const params = new URLSearchParams();
  const compressed = compressState(state.answers);
  params.set('state', compressed);
  
  // Include temporary preference rating if it exists
  if (state.tempPreferenceRating) {
    params.set('temp_rating', state.tempPreferenceRating);
  }
  
  // Update URL without reloading the page
  window.history.replaceState(
    {},
    '',
    `${window.location.pathname}?${params.toString()}`
  );
}

function navigateToMeal() {
  // Changed to navigate to the new meal-preferences.html page
  const params = new URLSearchParams();
  if (state.answers.length > 0) {
    const compressed = compressState(state.answers);
    params.set('state', compressed);
  }
  window.location.href = `meal-preferences.html?${params.toString()}`;
}

function navigateToMealDetails() {
  const params = new URLSearchParams();
  
  // Include existing answers
  if (state.answers.length > 0) {
    const compressed = compressState(state.answers);
    params.set('state', compressed);
  }
  
  // Pass temporary preference rating
  if (state.tempPreferenceRating) {
    params.set('temp_rating', state.tempPreferenceRating);
  }
  
  window.location.href = `meal-details.html?${params.toString()}`;
}

function navigateToThanks() {
  window.location.href = 'thanks.html';
}

/***********************************************************
 * CONTENT LOADING
 ***********************************************************/
async function loadMetadata(questionKey) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) {
    throw new Error(`Question key not found: ${questionKey}`);
  }
  
  try {
    const contentPath = getContentPath(questionKey);
    
    const response = await fetch(`${contentPath}/metadata.json`);
    if (!response.ok) {
      throw new Error(`Failed to load metadata for ${questionKey}: ${response.status} ${response.statusText}`);
    }
    
    const text = await response.text();
    if (!text.trim()) {
      throw new Error(`Empty metadata file for ${questionKey}`);
    }
    
    try {
      return JSON.parse(text);
    } catch (parseError) {
      throw new Error(`Invalid JSON in metadata for ${questionKey}: ${parseError.message}`);
    }
  } catch (error) {
    console.error(`Error loading metadata for ${questionKey}:`, error);
    throw error;
  }
}

async function loadPredictions(questionKey) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) {
    throw new Error(`Question key not found: ${questionKey}`);
  }
  
  try {
    const contentPath = getContentPath(questionKey);

    // Special handling for occlusion questions
    if (questionKey.startsWith('look_') || questionKey.startsWith('block_')) {
      // For occlusion questions, we should use the standard path      
      const response = await fetch(`${contentPath}/prediction.json`);
      if (!response.ok) {
        throw new Error(`Failed to load occlusion prediction for ${questionKey}: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      return data;
    }

    // For all other questions, use text predictions
    const response = await fetch(`${contentPath}/prediction.txt`);
    if (!response.ok) {
      throw new Error(`Failed to load prediction for ${questionKey}: ${response.status} ${response.statusText}`);
    }
    const text = await response.text();
    if (!text.trim()) {
      throw new Error(`Empty prediction file for ${questionKey}`);
    }
    return text;
  
  } catch (error) {
    console.error(`Error loading predictions for ${questionKey}:`, error);
    throw error;
  }
}

async function loadCurrentMealInfo() {
  try {
    // Use getContentPath to get the correct path for occlusion questions
    const occlusionPath = getContentPath('look_forward');
    const metadataPath = getContentPath('bite_order');
    
    // Load metadata
    const metadataResponse = await fetch(`${metadataPath}/metadata.json`);
    if (!metadataResponse.ok) {
      throw new Error(`Failed to load meal metadata: ${metadataResponse.status} ${metadataResponse.statusText}`);
    }
    const metadata = await metadataResponse.json();
    
    // Verify the image exists
    const imageResponse = await fetch(`${occlusionPath}/bite_occlusion_image.png`, { method: 'HEAD' });
    if (!imageResponse.ok) {
      throw new Error(`Failed to load meal image: ${imageResponse.status} ${imageResponse.statusText}`);
    }
    
    // Create a descriptive meal context
    const foodItems = metadata.food_items.join(' and ');
    const dips = metadata.dips.join(' and ');
    const context = metadata.context.replace('_', ' ');
    const tableType = metadata.table_type.replace('_', ' ');
    
    const description = `Imagine you are having a meal in a <span class="context">${context}</span> setting at a <span class="table-type">${tableType}</span>. 
    On your plate, you have <span class="food-items">${foodItems}</span>${dips ? ` with <span class="dips">${dips}</span> for dipping` : ''}. 
    Please answer the following questions about this meal scenario.`;
    
    state.currentMeal = {
      title: `Meal ${state.answers.length + 1} of 5`,
      image: `${occlusionPath}/bite_occlusion_image.png`,
      description: description
    };
    
  } catch (error) {
    console.error('Error loading meal info:', error);
    throw error;
  }
}

/**
 * Display an error message on the page
 * @param {string} message - The error message to display
 */
function showErrorMessage(message) {
  // Clear existing content
  const mealScreen = document.getElementById("meal-screen");
  if (mealScreen) {
    mealScreen.innerHTML = "";
  }

  // Create error container
  const errorContainer = document.createElement("div");
  errorContainer.style.backgroundColor = "#ffebee";
  errorContainer.style.border = "1px solid #f44336";
  errorContainer.style.borderRadius = "4px";
  errorContainer.style.padding = "1.5rem";
  errorContainer.style.margin = "2rem auto";
  errorContainer.style.maxWidth = "800px";
  errorContainer.style.boxShadow = "0 2px 4px rgba(0,0,0,0.1)";
  errorContainer.style.textAlign = "center";

  // Create error icon
  const errorIcon = document.createElement("div");
  errorIcon.innerHTML = "⚠️";
  errorIcon.style.fontSize = "3rem";
  errorIcon.style.marginBottom = "1rem";
  errorContainer.appendChild(errorIcon);

  // Create error title
  const errorTitle = document.createElement("h2");
  errorTitle.textContent = "Error Loading Content";
  errorTitle.style.marginBottom = "1rem";
  errorTitle.style.color = "#d32f2f";
  errorContainer.appendChild(errorTitle);

  // Create error message
  const errorMessage = document.createElement("p");
  errorMessage.textContent = message;
  errorMessage.style.fontSize = "1.1rem";
  errorMessage.style.marginBottom = "1.5rem";
  errorContainer.appendChild(errorMessage);

  // Create refresh button
  const refreshButton = document.createElement("button");
  refreshButton.textContent = "Try Again";
  refreshButton.style.padding = "0.75rem 1.5rem";
  refreshButton.style.backgroundColor = "#f44336";
  refreshButton.style.color = "white";
  refreshButton.style.border = "none";
  refreshButton.style.borderRadius = "4px";
  refreshButton.style.fontSize = "1rem";
  refreshButton.style.cursor = "pointer";
  refreshButton.onclick = () => window.location.reload();
  errorContainer.appendChild(refreshButton);

  // Add to page
  if (mealScreen) {
    mealScreen.appendChild(errorContainer);
  } else {
    document.body.appendChild(errorContainer);
  }

  // Hide next button
  const nextBtn = document.getElementById("next-btn");
  if (nextBtn) {
    nextBtn.style.display = "none";
  }
}

/***********************************************************
 * UI RENDERING
 ***********************************************************/
async function loadCurrentContent() {
  try {
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
  } catch (error) {
    console.error("Failed to load content:", error);
    showErrorMessage(error.message || "Failed to load content. Please try again.");
    throw error; // Re-throw to stop further processing
  }
}

async function getOptionsForQuestion(questionKey) {
  // For verbal and occlusion questions, always use Yes/No
  if (questionKey === 'verbal') {
    return ['Yes', 'No'];
  }

  if (questionKey.startsWith('look_') || questionKey.startsWith('block_')) {
    return ['Yes', 'No'];
  }
  
  // For all other questions, use metadata choices if available
  const metadata = state.metadata[questionKey];
  if (metadata && metadata.choices) {
    return metadata.choices;
  }
  
  // Fall back to initial options if no metadata
  return INITIAL_OPTIONS[questionKey];
}

/**
 * Checks if left-side occlusion is enabled for the current meal
 * @returns {boolean} True if left-side occlusion is enabled, false otherwise
 * @throws {Error} If metadata fails to load
 */
async function isLeftOcclusionEnabled() {
  try {
    // Get the content path for occlusion
    const contentPath = getContentPath('look_forward');
    
    // Directly load the metadata for occlusion
    const response = await fetch(`${contentPath}/metadata.json`);
    if (!response.ok) {
      throw new Error(`Failed to load occlusion metadata: ${response.status} ${response.statusText}`);
    }
    
    const text = await response.text();
    if (!text.trim()) {
      throw new Error('Empty occlusion metadata file');
    }
    
    const metadata = JSON.parse(text);
    
    // Check if the metadata includes left occlusion
    // We'll look for the 'occlusion_options' property that should include 'left'
    if (metadata.choices && metadata.choices.includes('left')) {
      return true;
    }
    
    return false;
  } catch (error) {
    throw error;
  }
}

// Add a global optionValues object to store values for each option
const optionValues = {
  optionA: {},
  optionB: {}
};

async function renderPredictionSummary() {
  // Create summary section
  const form = document.querySelector("#preferences-form, #questions-form"); // Work with either form
  const summarySection = document.createElement("div");
  summarySection.className = "prediction-summary";
  summarySection.style.marginBottom = "2rem";
  summarySection.style.backgroundColor = "#ffffff";
  summarySection.style.border = "1px solid #e0e0e0";
  summarySection.style.borderRadius = "8px";
  summarySection.style.padding = "1.5rem";
  summarySection.style.boxShadow = "0 2px 4px rgba(0,0,0,0.05)";
  
  const title = document.createElement("h3");
  title.textContent = "Robot Decision Summary";
  title.style.marginTop = "0";
  title.style.marginBottom = "0.5rem";
  summarySection.appendChild(title);
  
  const description = document.createElement("p");
  // Change description based on whether this is the first meal
  if (state.answers.length === 0) {
    description.innerHTML = "The robot is providing options for your first meal. Please compare Option A and Option B:";
  } else {
    description.innerHTML = "Based on your previous preferences, here are two possible sets of choices that the robot might make for your next meal. Please compare Option A and Option B:";
  }
  description.style.marginBottom = "1rem";
  summarySection.appendChild(description);
  
  // Create table for predictions
  const table = document.createElement("table");
  table.style.width = "100%";
  table.style.borderCollapse = "collapse";
  
  // Add table header
  const thead = document.createElement("thead");
  thead.style.backgroundColor = "#f0f0f0";
  const headerRow = document.createElement("tr");
  
  const questionHeader = document.createElement("th");
  questionHeader.textContent = "Setting";
  questionHeader.style.padding = "0.75rem";
  questionHeader.style.textAlign = "left";
  headerRow.appendChild(questionHeader);
  
  const optionAHeader = document.createElement("th");
  optionAHeader.textContent = "Option A";
  optionAHeader.style.padding = "0.75rem";
  optionAHeader.style.textAlign = "left";
  headerRow.appendChild(optionAHeader);
  
  const optionBHeader = document.createElement("th");
  optionBHeader.textContent = "Option B";
  optionBHeader.style.padding = "0.75rem";
  optionBHeader.style.textAlign = "left";
  headerRow.appendChild(optionBHeader);
  
  thead.appendChild(headerRow);
  table.appendChild(thead);
  
  // Randomize for each meal independently
  // Create a new randomization for this meal
  const isOptionAPersonalized = Math.random() < 0.5;
  state.optionMappings[state.answers.length] = isOptionAPersonalized;
  
  // Add table body
  const tbody = document.createElement("tbody");
  
  // Check if left occlusion should be shown
  const showLeftOcclusion = await isLeftOcclusionEnabled();

  // Filter out occlusion-related questions from the summary table
  // But keep all questions for the details page
  const questionsToShow = QUESTIONS.filter(q => !q.key.startsWith('look_') && !q.key.startsWith('block_'));
  
  // Clear option values for this meal
  optionValues.optionA = {};
  optionValues.optionB = {};
  
  // Still process all questions so option values are populated correctly
  for (const question of QUESTIONS) {
    const options = await getOptionsForQuestion(question.key);
    // Get the default option (first in the list)
    if (question.key.startsWith('look_') || question.key.startsWith('block_')) {
      defaultOption = "No";
    } else {
      defaultOption = options.length > 0 ? options[0] : "None";
    }
    
    // Get the personalized prediction
    let prediction = state.predictions[question.key];
    if (Array.isArray(prediction)) {
      prediction = prediction[0];
    }
    
    // Format prediction for display
    let formattedPrediction = prediction;
    
    // Handle Yes/No predictions for verbal and occlusion questions
    if (question.key === 'verbal' && prediction) {
      formattedPrediction = prediction.trim() === 'True' ? 'Yes' : 'No';
    }
    
    // Handle occlusion predictions
    if ((question.key.startsWith('look_') || question.key.startsWith('block_')) && prediction) {
      const direction = question.key.includes('forward') ? 'front' : 'left';
      const isLooking = question.key.startsWith('look_');
      
      if (prediction.relevant_pois && prediction.occluded_pois) {
        // Use the prediction data directly from this question
        if (isLooking) {
          formattedPrediction = prediction.relevant_pois.includes(direction) ? 'Yes' : 'No';
        } else {
          formattedPrediction = prediction.occluded_pois.includes(direction) ? 'Yes' : 'No';
        }
      } else {
        // Fallback defaults if prediction data is missing
        if (state.answers.length === 0) {
          if (isLooking) {
            // Default to looking forward only
            formattedPrediction = direction === 'front' ? 'Yes' : 'No';
          } else {
            // Default to no occlusion
            formattedPrediction = 'No';
          }
        } else {
          formattedPrediction = isLooking && direction === 'front' ? 'Yes' : 'No';
        }
      }
    }
    
    // Determine which option goes in which column based on current meal's randomization
    const optionA = isOptionAPersonalized ? formattedPrediction : defaultOption;
    const optionB = isOptionAPersonalized ? defaultOption : formattedPrediction;
    
    // Store these values in our global option values map - for ALL questions
    optionValues.optionA[question.key] = optionA;
    optionValues.optionB[question.key] = optionB;
    
    // Skip rendering occlusion questions in the table
    if (question.key.startsWith('look_') || question.key.startsWith('block_')) {
      continue;
    }
    
    const row = document.createElement("tr");
    row.style.borderBottom = "1px solid #eee";
    
    const questionCell = document.createElement("td");
    questionCell.textContent = question.text;
    questionCell.style.padding = "0.75rem";
    row.appendChild(questionCell);
    
    const optionACell = document.createElement("td");
    optionACell.textContent = optionA || "No option";
    optionACell.style.padding = "0.75rem";
    
    const optionBCell = document.createElement("td");
    optionBCell.textContent = optionB || "No option";
    optionBCell.style.padding = "0.75rem";
    
    // Highlight both cells if the options are different
    if (optionA !== optionB) {
      optionACell.style.fontWeight = "bold";
      optionACell.style.color = "#1a73e8";
      
      optionBCell.style.fontWeight = "bold";
      optionBCell.style.color = "#1a73e8";
    }
    
    row.appendChild(optionACell);
    row.appendChild(optionBCell);
    
    tbody.appendChild(row);
  }
  
  table.appendChild(tbody);
  summarySection.appendChild(table);
  
  // Add a divider between the table and images
  const divider = document.createElement("hr");
  divider.style.margin = "2rem 0";
  divider.style.border = "none";
  divider.style.borderTop = "1px solid #e0e0e0";
  summarySection.appendChild(divider);
  
  // Replace the visual heading with the requested text
  const visualText = document.createElement("p");
  visualText.textContent = "The robot is also considering two different configurations:";
  visualText.style.marginBottom = "1.5rem";
  visualText.style.textAlign = "center";
  visualText.style.fontSize = "1.1rem";
  visualText.style.fontWeight = "500";
  visualText.style.color = "#333";
  summarySection.appendChild(visualText);
  
  // Create container for the option images
  const optionImagesContainer = document.createElement("div");
  optionImagesContainer.style.display = "flex";
  optionImagesContainer.style.justifyContent = "space-between";
  optionImagesContainer.style.gap = "2rem";
  optionImagesContainer.style.margin = "0 0 1rem 0";
  summarySection.appendChild(optionImagesContainer);
  
  // Option A Image Container
  const optionAContainer = document.createElement("div");
  optionAContainer.style.flex = "1";
  optionAContainer.style.textAlign = "center";
  
  // Option A Heading
  const optionAHeading = document.createElement("h4");
  optionAHeading.textContent = "Option A";
  optionAHeading.style.marginBottom = "1rem";
  optionAHeading.style.color = "#1a73e8";
  optionAContainer.appendChild(optionAHeading);
  
  // Option A Image
  const optionAImage = document.createElement("img");
  optionAImage.src = isOptionAPersonalized ? state.currentMeal.image : `content/default_meal_images/meal${state.answers.length + 1}.png`;
  optionAImage.alt = "Option A Preview";
  optionAImage.style.width = "100%";
  optionAImage.style.maxWidth = "280px";
  optionAImage.style.border = "2px solid #1a73e8";
  optionAImage.style.borderRadius = "8px";
  optionAContainer.appendChild(optionAImage);
  
  // Option B Image Container
  const optionBContainer = document.createElement("div");
  optionBContainer.style.flex = "1";
  optionBContainer.style.textAlign = "center";
  
  // Option B Heading
  const optionBHeading = document.createElement("h4");
  optionBHeading.textContent = "Option B";
  optionBHeading.style.marginBottom = "1rem";
  optionBHeading.style.color = "#1a73e8";
  optionBContainer.appendChild(optionBHeading);
  
  // Option B Image
  const optionBImage = document.createElement("img");
  optionBImage.src = isOptionAPersonalized ? `content/default_meal_images/meal${state.answers.length + 1}.png` : state.currentMeal.image;
  optionBImage.alt = "Option B Preview";
  optionBImage.style.width = "100%";
  optionBImage.style.maxWidth = "280px";
  optionBImage.style.border = "2px solid #1a73e8";
  optionBImage.style.borderRadius = "8px";
  optionBContainer.appendChild(optionBImage);
  
  // Add both option containers to the main container
  optionImagesContainer.appendChild(optionAContainer);
  optionImagesContainer.appendChild(optionBContainer);
  
  // Insert summary section at the beginning of the form
  form.insertBefore(summarySection, form.firstChild);
}

async function showMealPreferences() {
  try {
    // Load content before showing the meal
    await loadCurrentContent();
    
    const mealTitle = document.getElementById("meal-title");
    
    mealTitle.textContent = state.currentMeal.title;
    
    // Create a styled container for the meal description
    const descContainer = document.createElement("div");
    descContainer.style.backgroundColor = "#ffffff";
    descContainer.style.border = "1px solid #e0e0e0";
    descContainer.style.borderRadius = "6px";
    descContainer.style.padding = "1.25rem";
    descContainer.style.marginTop = "1rem";
    descContainer.style.marginBottom = "1.5rem";
    descContainer.style.boxShadow = "0 2px 4px rgba(0,0,0,0.05)";
    
    // Add heading for the scenario
    const scenarioHeading = document.createElement("h3");
    scenarioHeading.textContent = "Meal Scenario";
    scenarioHeading.style.marginTop = "0";
    scenarioHeading.style.marginBottom = "0.75rem";
    scenarioHeading.style.color = "#333";
    descContainer.appendChild(scenarioHeading);
    
    // Create the meal description element
    const mealDesc = document.createElement("p");
    mealDesc.id = "meal-desc";
    mealDesc.innerHTML = state.currentMeal.description;
    mealDesc.className = 'meal-context';
    mealDesc.style.margin = "0";
    mealDesc.style.lineHeight = "1.5";
    mealDesc.style.fontSize = "1.05rem";
    
    // Add the description to the container
    descContainer.appendChild(mealDesc);
    
    // Insert the container after the title
    const mealSection = document.querySelector('section');
    mealTitle.parentNode.insertBefore(descContainer, mealTitle.nextSibling);
    
    // Add the form with placeholder for preference rating
    await renderPreferencesForm();
    
    // Add the summary table with images
    await renderPredictionSummary();
    
    // Finally, add the preference rating after the summary
    await addPreferenceRatingAfterImages();
  } catch (error) {
    console.error("Failed to show meal preferences:", error);
    showErrorMessage(error.message || "Failed to display meal preferences. Please try again.");
  }
}

async function showMealDetails() {
  try {
    // Load content before showing the meal details (reusing the same content)
    await loadCurrentContent();
    
    const mealTitle = document.getElementById("meal-title");
    const mealImg = document.getElementById("meal-img");
    
    mealTitle.textContent = state.currentMeal.title;
    
    // Set the image source but hide it initially - we'll reposition it later
    mealImg.src = state.currentMeal.image;
    mealImg.style.display = "none";
    
    // Create a styled container for the meal description
    const descContainer = document.createElement("div");
    descContainer.style.backgroundColor = "#ffffff";
    descContainer.style.border = "1px solid #e0e0e0";
    descContainer.style.borderRadius = "6px";
    descContainer.style.padding = "1.25rem";
    descContainer.style.marginTop = "1rem";
    descContainer.style.marginBottom = "1.5rem";
    descContainer.style.boxShadow = "0 2px 4px rgba(0,0,0,0.05)";
    
    // Add heading for the scenario
    const scenarioHeading = document.createElement("h3");
    scenarioHeading.textContent = "Recall the Meal Scenario";
    scenarioHeading.style.marginTop = "0";
    scenarioHeading.style.marginBottom = "0.75rem";
    scenarioHeading.style.color = "#333";
    descContainer.appendChild(scenarioHeading);
    
    // Create the meal description element
    const mealDesc = document.createElement("p");
    mealDesc.id = "meal-desc";
    mealDesc.innerHTML = state.currentMeal.description;
    mealDesc.className = 'meal-context';
    mealDesc.style.margin = "0";
    mealDesc.style.lineHeight = "1.5";
    mealDesc.style.fontSize = "1.05rem";
    
    // Add the description to the container
    descContainer.appendChild(mealDesc);
    
    // Render the form with only the detailed questions
    await renderDetailsForm(descContainer, mealImg);
  } catch (error) {
    console.error("Failed to show meal details:", error);
    showErrorMessage(error.message || "Failed to display meal details. Please try again.");
  }
}

// Function to create just the comparison table and preference rating
async function renderPreferencesForm() {
  const form = document.getElementById("preferences-form");
  form.innerHTML = ""; // Clear existing content
  
  // We'll add the preference rating after the renderPredictionSummary and image options are added
  // This creates a placeholder div that we'll populate later
  const preferenceRatingPlaceholder = document.createElement("div");
  preferenceRatingPlaceholder.id = "preference-rating-container";
  form.appendChild(preferenceRatingPlaceholder);
  
  // Check if we have a saved temporary preference rating
  if (state.tempPreferenceRating) {
    const ratingSelect = form.querySelector('select[name="preference_rating"]');
    if (ratingSelect) {
      ratingSelect.value = state.tempPreferenceRating;
      checkPreferenceFormCompletion();
    }
  }
}

// Updated function to add preference rating in the correct position
async function addPreferenceRatingAfterImages() {
  // Find the placeholder
  const ratingContainer = document.getElementById("preference-rating-container");
  if (!ratingContainer) return;
  
  // Add preference rating question
  const ratingDiv = document.createElement("div");
  ratingDiv.className = "preference-rating";
  ratingDiv.style.marginTop = "2.5rem";
  ratingDiv.style.marginBottom = "2.5rem";
  ratingDiv.style.padding = "1.25rem";
  ratingDiv.style.border = "1px solid #d0d0d0";
  ratingDiv.style.borderRadius = "6px";
  ratingDiv.style.backgroundColor = "#f5f8ff";
  ratingDiv.style.boxShadow = "0 2px 4px rgba(0,0,0,0.05)";
  
  // Create title for the rating section
  const ratingTitle = document.createElement("h3");
  ratingTitle.style.marginTop = "0";
  ratingTitle.style.marginBottom = "0.75rem";
  ratingTitle.style.fontSize = "1.25rem";
  
  let ratingText = "";
  ratingTitle.textContent = "Compare Options";
  ratingText = "To what extent do you prefer Option A or Option B for this meal?";
  
  ratingDiv.appendChild(ratingTitle);
  
  const ratingLabel = document.createElement("p");
  ratingLabel.htmlFor = "preference_rating";
  ratingLabel.innerHTML = ratingText;
  ratingLabel.className = "meal-context";
  ratingLabel.style.fontSize = "1.05rem";
  ratingLabel.style.marginBottom = "1rem";
  
  // Create rating select dropdown
  const ratingSelect = document.createElement("select");
  ratingSelect.id = "preference_rating";
  ratingSelect.name = "preference_rating";
  ratingSelect.style.width = "100%";
  ratingSelect.style.padding = "0.75rem";
  ratingSelect.style.marginTop = "0.5rem";
  ratingSelect.style.fontSize = "1.1rem";
  ratingSelect.style.border = "1px solid #ccc";
  ratingSelect.style.borderRadius = "4px";
  ratingSelect.style.backgroundColor = "#fff";

  // Define the descriptive rating options
  const ratingOptions = [
    { value: "1", text: "1: Strongly prefer Option A" },
    { value: "2", text: "2: Prefer Option A" },
    { value: "3", text: "3: Somewhat prefer Option A" },
    { value: "4", text: "4: Neutral" },
    { value: "5", text: "5: Somewhat prefer Option B" },
    { value: "6", text: "6: Prefer Option B" },
    { value: "7", text: "7: Strongly prefer Option B" }
  ];

  // Add empty default option
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "Select your preference...";
  ratingSelect.appendChild(defaultOption);

  // Add all rating options with descriptive text
  ratingOptions.forEach(optionData => {
    const option = document.createElement("option");
    option.value = optionData.value;
    option.textContent = optionData.text;
    ratingSelect.appendChild(option);
  });
  
  ratingSelect.addEventListener("change", checkPreferenceFormCompletion);
  
  ratingDiv.appendChild(ratingLabel);
  ratingDiv.appendChild(ratingSelect);
  
  // Replace the placeholder with the actual rating div
  ratingContainer.parentNode.replaceChild(ratingDiv, ratingContainer);
  
  // Set value if we have a saved temp rating
  if (state.tempPreferenceRating) {
    ratingSelect.value = state.tempPreferenceRating;
    checkPreferenceFormCompletion();
  }
}

// Function to create just the detailed preference questions form
async function renderDetailsForm(descContainer, mealImg) {
  const form = document.getElementById("questions-form");
  form.innerHTML = ""; // Clear existing questions
  
  // Add section heading for meal questions (STEP 1)
  const sectionHeading = document.createElement("h3");
  sectionHeading.textContent = "Help the robot learn your preferences";
  sectionHeading.style.marginTop = "0.5rem";
  sectionHeading.style.marginBottom = "0.25rem";
  form.appendChild(sectionHeading);
  
  // Add section description
  const sectionDesc = document.createElement("p");
  sectionDesc.innerHTML = state.answers.length === 0
    ? "For your first meal, please select your preferences for each setting:"
    : "Please review the robot's personalized choices and adjust if needed:";
  sectionDesc.style.marginBottom = "1.5rem";
  form.appendChild(sectionDesc);
  
  // Add meal scenario section (STEP 2)
  form.appendChild(descContainer);
  
  // Check if left occlusion should be shown
  const showLeftOcclusion = await isLeftOcclusionEnabled();

  // Filter questions if left occlusion is disabled
  const questionsToShow = showLeftOcclusion ? 
    QUESTIONS : 
    QUESTIONS.filter(q => !q.key.includes('left'));
  
  // Create containers for non-occlusion and occlusion questions
  const nonOcclusionContainer = document.createElement("div");
  nonOcclusionContainer.className = "non-occlusion-questions";
  nonOcclusionContainer.style.marginBottom = "1.5rem";
  
  const occlusionContainer = document.createElement("div");
  occlusionContainer.className = "occlusion-questions";
  occlusionContainer.style.marginTop = "1.5rem";
  
  // Sort questions into the appropriate containers
  for (const question of questionsToShow) {
    const options = await getOptionsForQuestion(question.key);
    
    const wrapper = document.createElement("div");
    wrapper.style.marginBottom = "0.5rem";
    wrapper.style.padding = "0.75rem";
    wrapper.style.border = "1px solid #e0e0e0";
    wrapper.style.borderRadius = "4px";
    wrapper.style.backgroundColor = "#fafafa";
    
    const label = document.createElement("label");
    label.setAttribute("for", question.key);
    label.className = 'meal-context';
    label.style.fontWeight = "500";
    label.style.display = "block";
    label.style.marginBottom = "0.75rem";
    
    // Get the prediction for this question
    let prediction = state.predictions[question.key];
    if (Array.isArray(prediction)) {
      prediction = prediction[0];
    }
    
    // Handle Yes/No predictions for verbal and occlusion questions
    if (question.key === 'verbal' && prediction) {
      prediction = prediction.trim() === 'True' ? 'Yes' : 'No';
    }
    
    // Handle occlusion predictions
    if ((question.key.startsWith('look_') || question.key.startsWith('block_')) && prediction) {
      const direction = question.key.includes('forward') ? 'front' : 'left';
      const isLooking = question.key.startsWith('look_');
      
      if (prediction.relevant_pois && prediction.occluded_pois) {
        // Use the prediction data directly from this question
        if (isLooking) {
          prediction = prediction.relevant_pois.includes(direction) ? 'Yes' : 'No';
        } else {
          prediction = prediction.occluded_pois.includes(direction) ? 'Yes' : 'No';
        }
      } else {
        // Fallback defaults if prediction data is missing
        if (state.answers.length === 0) {
          if (isLooking) {
            // Default to looking forward only
            prediction = direction === 'front' ? 'Yes' : 'No';
          } else {
            // Default to no occlusion
            prediction = 'No';
          }
        } else {
          prediction = isLooking && direction === 'front' ? 'Yes' : 'No';
        }
      }
    }
    
    // Create the question text based on the type
    if (question.key === 'bite_order') {
      label.innerHTML = `How would you like your <span class="context">bites served?</span>`;
    } else if (question.key === 'ready_signal') {
      label.innerHTML = `What would you prefer to use as a <span class="context">ready signal?</span>`;
    } else if (question.key === 'verbal') {
      label.innerHTML = `Would you like the robot to be <span class="context">verbal</span> during this meal?`;
    } else if (question.key.startsWith('look_')) {
      if (question.key === 'look_left') {
        label.innerHTML = `Would you typically be looking <span class="context">slightly left</span> during this meal?`;
      } else {
        label.innerHTML = `Would you typically be looking <span class="context">forward</span> during this meal?`;
      }
    } else if (question.key.startsWith('block_')) {
      if (question.key === 'block_left') {
        label.innerHTML = `Is the robot uncomfortably blocking your <span class="context">leftward</span> sight?`;
      } else {
        label.innerHTML = `Is the robot uncomfortably blocking your <span class="context">forward</span> sight?`;
      }
    } else {
      label.textContent = question.text;
    }
    
    const select = document.createElement("select");
    select.id = question.key;
    select.name = question.key;
    select.style.width = "100%";
    select.style.padding = "0.5rem";
    select.style.fontSize = "1rem";
    select.style.borderRadius = "4px";
    select.style.border = "1px solid #ccc";
    
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
    }
    
    // If we have metadata for this question, add it as a data attribute
    const metadata = state.metadata[question.key];
    if (metadata) {
      select.dataset.metadata = JSON.stringify(metadata);
    }
    
    // Pre-select the value from the selected option
    const isOptionAPersonalized = state.optionMappings[state.answers.length] || false;
    const selectedOptionData = state.tempPreferenceRating ? 
      (state.tempPreferenceRating <= 3 ? optionValues.optionA : 
       state.tempPreferenceRating >= 5 ? optionValues.optionB : 
       null) : null;
    
    // If user has a preference, pre-select values based on that preference
    if (selectedOptionData && selectedOptionData[question.key]) {
      select.value = selectedOptionData[question.key];
    }
    
    select.addEventListener("change", checkDetailsFormCompletion);
    
    // Add special handling for "looking" questions to disable corresponding "blocking" questions
    if (question.key.startsWith('look_')) {
      select.addEventListener("change", (e) => {
        const direction = question.key.includes('forward') ? 'forward' : 'left';
        const blockingQuestionId = `block_${direction}`;
        const blockingSelect = document.getElementById(blockingQuestionId);
        
        if (blockingSelect) {
          if (e.target.value === 'No') {
            // If not looking in this direction, can't be uncomfortably blocking
            blockingSelect.value = 'No';
            blockingSelect.disabled = true;
            blockingSelect.parentNode.style.opacity = '0.6';
            // Trigger validation check after setting the value
            checkDetailsFormCompletion();
          } else {
            // Re-enable the blocking question
            blockingSelect.disabled = false;
            blockingSelect.parentNode.style.opacity = '1';
          }
        }
      });
    }
    
    wrapper.appendChild(label);
    wrapper.appendChild(select);
    
    // Add the question to the appropriate container
    if (question.key.startsWith('look_') || question.key.startsWith('block_')) {
      occlusionContainer.appendChild(wrapper);
    } else {
      nonOcclusionContainer.appendChild(wrapper);
    }
  }
  
  // STEP 3: Add non-occlusion questions to the form
  form.appendChild(nonOcclusionContainer);
  
  // STEP 4: Add the meal image
  mealImg.style.display = "block";
  mealImg.style.maxWidth = "100%";
  mealImg.style.marginTop = "0.75rem";
  mealImg.style.marginBottom = "1rem";
  mealImg.style.border = "1px solid #ddd";
  mealImg.style.borderRadius = "8px";
  form.appendChild(mealImg);
  
  // STEP 5: Add occlusion questions
  // Add instructions for occlusion questions
  if (occlusionContainer.children.length > 0) {
    const occlusionInstructions = document.createElement("p");
    occlusionInstructions.textContent = "Refer to the image above to answer the following questions:";
    occlusionInstructions.style.marginTop = "0.5rem";
    occlusionInstructions.style.marginBottom = "0.5rem";
    occlusionInstructions.style.fontStyle = "italic";
    occlusionInstructions.style.color = "#555";
    form.appendChild(occlusionInstructions);
    form.appendChild(occlusionContainer);
    
    // Trigger change events for looking questions to initialize blocking questions properly
    const lookingSelects = form.querySelectorAll('select[id^="look_"]');
    lookingSelects.forEach(select => {
      const event = new Event('change');
      select.dispatchEvent(event);
    });
  }
  
  // Check form completion status after all fields are added
  checkDetailsFormCompletion();
}

// Form validation for the preference page
function checkPreferenceFormCompletion() {
  const form = document.getElementById("preferences-form");
  const nextBtn = document.getElementById("next-btn");
  const preferenceRating = form.querySelector('select[name="preference_rating"]');
  
  // Check if preference rating has been selected
  const isComplete = preferenceRating && preferenceRating.value;
  
  nextBtn.disabled = !isComplete;
}

// Form validation for the details page
function checkDetailsFormCompletion() {
  const form = document.getElementById("questions-form");
  const nextBtn = document.getElementById("next-btn");
  const selects = form.querySelectorAll("select");
  
  // Check if all selects have a value
  let allAnswered = true;
  
  selects.forEach(select => {
    // Consider a select answered if it has a value OR it's disabled (with a value)
    if (!select.value && !select.disabled) {
      allAnswered = false;
    }
  });
  
  nextBtn.disabled = !allAnswered;
}

// Handler for the next button on preferences page
async function handlePreferencesNextClick() {
  const form = document.getElementById("preferences-form");
  const preferenceRating = form.querySelector('select[name="preference_rating"]');
  
  if (preferenceRating && preferenceRating.value) {
    // Store the preference rating temporarily
    state.tempPreferenceRating = preferenceRating.value;
    
    // Update URL to include the temporary preference
    updateUrlState();
    
    // Navigate to the details page
    navigateToMealDetails();
  }
}

// Handler for the next button on details page
async function handleDetailsNextClick() {
  // Collect answers from the details form
  const answers = await collectAnswers();
  
  // Add the temporary preference rating to answers
  if (state.tempPreferenceRating) {
    answers.preference_rating = {
      value: state.tempPreferenceRating,
      metadata: null
    };
    
    // Clear the temporary preference rating
    state.tempPreferenceRating = null;
  }
  
  // Store which option is personalized to interpret ratings correctly
  answers.isOptionAPersonalized = state.optionMappings[state.answers.length];
  
  // Save answers and proceed
  state.answers.push(answers);
  
  if (state.answers.length < 5) {
    navigateToMeal();
  } else {
    // Send final state to Google Form before navigating to thanks page
    await sendToGoogleForm();
    navigateToThanks();
  }
}

async function collectAnswers() {
  const form = document.getElementById("questions-form");
  const answers = {};
  
  // Note: We don't collect preference_rating here anymore as it's passed in from the previous page
  
  // Check if left occlusion is enabled
  const showLeftOcclusion = await isLeftOcclusionEnabled();
  
  // Filter questions if left occlusion is disabled
  const questionsToCollect = showLeftOcclusion ? 
    QUESTIONS : 
    QUESTIONS.filter(q => !q.key.includes('left'));
  
  // Initialize occlusion structure if any occlusion questions exist
  if (questionsToCollect.some(q => q.key.startsWith('look_') || q.key.startsWith('block_'))) {
    answers.occlusion = {
      relevant_pois: [],
      occluded_pois: []
    };
  }
  
  for (const question of questionsToCollect) {
    const select = form.querySelector(`select[name="${question.key}"]`);
    if (!select) continue; // Skip if the select doesn't exist (might be filtered out)
    
    const value = select.value;
    
    // Handle special cases for storing answers
    if (question.key === 'verbal') {
      // Convert Yes/No back to True/False
      answers[question.key] = {
        value: value === 'Yes' ? 'True' : 'False',
        metadata: select.dataset.metadata ? JSON.parse(select.dataset.metadata) : null
      };
    } 
    
    else if (question.key.startsWith('look_') || question.key.startsWith('block_')) {
      // For occlusion questions, we need to update the prediction.json structure
      const direction = question.key.includes('forward') ? 'front' : 'left';
      const isLooking = question.key.startsWith('look_');
      
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
  }
  
  // If left occlusion is disabled, explicitly set left-related values to null
  // This ensures consistency in the data structure
  if (!showLeftOcclusion) {
    answers.look_left = {
      value: null,
      metadata: null
    };
    
    answers.block_left = {
      value: null,
      metadata: null
    };
  }
  
  return answers;
}

/**
 * Sends the final state data to a Google Form
 */
async function sendToGoogleForm() {
  try {
    // Replace with your actual Google Form URL
    // The URL should be the form's "formResponse" endpoint
    const googleFormUrl = "https://docs.google.com/forms/d/e/1FAIpQLSfLnBKpCIQQxVQ44__VRmmsQveIIDsWFEgBioKYcoFBqh2KOA/formResponse";
    
    // Prepare the data to send
    // You need to use the actual field names from your Google Form
    // These are typically named "entry.XXXXXXX" where X is a number
    const formData = new FormData();
    
    // Add the compressed state as a single field
    // Replace "entry.XXXXXXX" with your actual form field ID
    formData.append("entry.437529290", compressState(state.answers));
    
    // Add a summary of option mappings for easier analysis
    // This creates a string like "1:A,2:B,3:A,4:A,5:B" where A/B means personalized was option A or B
    const mappingSummary = state.optionMappings
      .map((isA, index) => `${index + 1}:${isA ? 'A' : 'B'}`)
      .join(',');
    formData.append("entry.437529291", mappingSummary);
    
    // Send the data using fetch API with POST method
    const response = await fetch(googleFormUrl, {
      method: "POST",
      mode: "no-cors", // Google Forms requires this
      body: formData
    });
    
    return true;
  } catch (error) {
    console.error("Error sending data to Google Form:", error);
    return false;
  }
}

// Add back a minimal redirect version of showMeal for backward compatibility
async function showMeal() {
  // This function just redirects to the new flow
  navigateToMeal();
}

// Add back a minimal version of checkFormCompletion for backward compatibility
function checkFormCompletion() {
  // Do nothing
}

/***********************************************************
 * INIT
 ***********************************************************/
// Initialize state from URL on page load
const { answers, tempPreferenceRating } = getStateFromUrl();
state.answers = answers;
state.tempPreferenceRating = tempPreferenceRating;

// Reconstruct option mappings from loaded answers if available
if (answers && answers.length > 0) {
  state.optionMappings = answers.map(answer => answer.isOptionAPersonalized);
}

// Set up event listeners based on current page
document.addEventListener('DOMContentLoaded', () => {
  // We don't set up event listeners here anymore as they are now set in each HTML file
  if (window.location.pathname.endsWith('index.html') || window.location.pathname.endsWith('/')) {
    // Handle the start button on the index page
    document.getElementById('start-btn').addEventListener('click', () => {
      navigateToMeal();
    });
  }
});

// Add browser back/forward button support
window.addEventListener('popstate', () => {
  const { answers, tempPreferenceRating } = getStateFromUrl();
  state.answers = answers;
  state.tempPreferenceRating = tempPreferenceRating;
  
  const path = window.location.pathname;
  if (path.endsWith('meal-preferences.html')) {
    showMealPreferences();
  } else if (path.endsWith('meal-details.html')) {
    showMealDetails();
  }
}); 