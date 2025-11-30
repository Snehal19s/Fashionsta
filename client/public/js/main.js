// client/public/js/main.js
const API_BASE_URL = 'http://localhost:5001/api'; // Your backend API port
const STATIC_BASE_URL = 'http://localhost:5001'; // Used for serving uploaded images

// --- Utility Functions ---
function getToken() {
    return localStorage.getItem('authToken');
}

function getUserInfo() {
    const userInfo = localStorage.getItem('userInfo');
    return userInfo ? JSON.parse(userInfo) : null;
}

async function makeApiRequest(endpoint, method = 'GET', body = null, requiresAuth = false) {
    const headers = {
        'Content-Type': 'application/json',
    };
    if (requiresAuth) {
        const token = getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        } else {
            console.warn('Auth token not found for protected route:', endpoint);
            // Optionally redirect to login or handle as an error
            // window.location.href = '/login.html';
            // return Promise.reject('Not authorized');
        }
    }

    const config = {
        method,
        headers,
    };

    if (body) {
        if (body instanceof FormData) { // For file uploads
            delete headers['Content-Type']; // Let browser set it for FormData
            config.body = body;
        } else {
            config.body = JSON.stringify(body);
        }
    }

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, config);
        if (!response.ok) {
            let errorData = { message: response.statusText };
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                errorData = await response.json().catch(() => ({ message: response.statusText }));
            }
            console.error(`API Error (${response.status}) on ${endpoint}:`, errorData.message);
            throw new Error(errorData.message || `Request failed with status ${response.status}`);
        }
        // For 204 No Content, response.json() will fail
        if (response.status === 204) {
            return null;
        }
        return await response.json();
    } catch (error) {
        console.error('Network or API request error:', error);
        throw error; // Re-throw to be caught by calling function
    }
}

function displayMessage(elementId, message, isError = false) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.className = isError ? 'form-message error' : 'form-message success';
        element.style.display = 'block';
        setTimeout(() => {
            element.style.display = 'none';
            element.textContent = '';
        }, 5000); // Hide after 5 seconds
    }
}

function showLoader(loaderId = 'page-loader') {
    const loader = document.getElementById(loaderId);
    if (loader) loader.classList.remove('d-none');
}

function hideLoader(loaderId = 'page-loader') {
    const loader = document.getElementById(loaderId);
    if (loader) loader.classList.add('d-none');
}


// --- UI Updates ---
function updateLoginStatusUI() {
    const userProfileLink = document.getElementById('user-profile-link');
    const adminLink = document.getElementById('admin-link');
    const logoutButton = document.getElementById('logout-button'); // Assuming you add a logout button
    const loginLink = document.getElementById('login-link'); // Your login link
    const signupLink = document.getElementById('signup-link'); // Your signup link

    const userInfo = getUserInfo();

    if (userInfo) {
        if (userProfileLink) userProfileLink.innerHTML = `<a href="profile.html">${userInfo.name}</a>`;
        if (adminLink && userInfo.isAdmin) adminLink.style.display = 'inline-block';
        if (logoutButton) logoutButton.style.display = 'inline-block';
        if (loginLink) loginLink.style.display = 'none';
        if (signupLink) signupLink.style.display = 'none';
    } else {
        if (userProfileLink) userProfileLink.innerHTML = '<a href="login.html">Login/Signup</a>';
        if (adminLink) adminLink.style.display = 'none';
        if (logoutButton) logoutButton.style.display = 'none';
        if (loginLink) loginLink.style.display = 'inline-block';
        if (signupLink) signupLink.style.display = 'inline-block';
    }
}

async function updateCartCount() {
    const cartCountEl = document.getElementById('cart-count');
    if (!cartCountEl || !getToken()) { // Only update if element exists and user is logged in
        if(cartCountEl) cartCountEl.textContent = '0';
        return;
    }
    try {
        const cart = await makeApiRequest('/users/cart', 'GET', null, true);
        cartCountEl.textContent = cart.reduce((sum, item) => sum + item.quantity, 0);
    } catch (error) {
        console.error('Failed to update cart count:', error);
        if(cartCountEl) cartCountEl.textContent = '0';
    }
}

async function updateWishlistCount() {
    const wishlistCountEl = document.getElementById('wishlist-count');
     if (!wishlistCountEl || !getToken()) { // Only update if element exists and user is logged in
        if(wishlistCountEl) wishlistCountEl.textContent = '0';
        return;
    }
    try {
        const wishlist = await makeApiRequest('/users/wishlist', 'GET', null, true);
        wishlistCountEl.textContent = wishlist.length;
    } catch (error) {
        console.error('Failed to update wishlist count:', error);
        if(wishlistCountEl) wishlistCountEl.textContent = '0';
    }
}

// --- Search Bar & Autocomplete ---
function setupSearch() {
    const searchBar = document.getElementById('search-bar');
    const autocompleteResults = document.getElementById('autocomplete-results');

    if (searchBar && autocompleteResults) {
        let debounceTimer;
        searchBar.addEventListener('input', async (e) => {
            const query = e.target.value.trim();
            clearTimeout(debounceTimer);

            if (query.length < 2) { // Min 2 chars to search
                autocompleteResults.innerHTML = '';
                autocompleteResults.style.display = 'none';
                return;
            }

            debounceTimer = setTimeout(async () => {
                try {
                    const results = await makeApiRequest(`/products/search/autocomplete?q=${encodeURIComponent(query)}`);
                    autocompleteResults.innerHTML = '';
                    if (results.length > 0) {
                        results.forEach(product => {
                            const div = document.createElement('div');
                            div.textContent = `${product.name} (in ${product.category})`;
                            div.onclick = () => {
                                window.location.href = `product-detail.html?id=${product._id}`;
                            };
                            autocompleteResults.appendChild(div);
                        });
                        autocompleteResults.style.display = 'block';
                    } else {
                        autocompleteResults.style.display = 'none';
                    }
                } catch (error) {
                    console.error('Autocomplete search failed:', error);
                    autocompleteResults.style.display = 'none';
                }
            }, 300); // Debounce time
        });

        // Hide autocomplete when clicking outside
        document.addEventListener('click', (e) => {
            if (!searchBar.contains(e.target) && !autocompleteResults.contains(e.target)) {
                autocompleteResults.style.display = 'none';
            }
        });

        // Handle search on enter
        searchBar.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const query = searchBar.value.trim();
                if (query) {
                    window.location.href = `products.html?keyword=${encodeURIComponent(query)}`;
                }
            }
        });
    }
}


// --- Document Ready / Initializations ---
document.addEventListener('DOMContentLoaded', () => {
    updateLoginStatusUI();
    if(getToken()){ // Only fetch counts if user is logged in.
      updateCartCount();
      updateWishlistCount();
    }
    setupSearch();

    // General logout button (if you add one to your nav or profile dropdown)
    const logoutButton = document.getElementById('logout-button');
    if (logoutButton) {
        logoutButton.addEventListener('click', () => {
            localStorage.removeItem('authToken');
            localStorage.removeItem('userInfo');
            updateLoginStatusUI();
            updateCartCount(); // Reset to 0
            updateWishlistCount(); // Reset to 0
            window.location.href = 'index.html';
        });
    }
});

// Assuming this code is part of your product detail or listing page where you show the product info
function displayProductInfo(product) {
    const nameElement = document.getElementById('product-name');
    const priceElement = document.getElementById('product-price');
    const descriptionElement = document.getElementById('product-description');
    const imageElement = document.getElementById('product-image');

    if (nameElement) nameElement.textContent = product.name;
    if (priceElement) priceElement.innerHTML = `₹${product.price.toFixed(2)}`;
    if (descriptionElement) descriptionElement.textContent = product.description;
    if (imageElement) imageElement.src = `${STATIC_BASE_URL}/uploads/${product.image}`;
}