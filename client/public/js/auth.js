// client/public/js/auth.js

document.addEventListener('DOMContentLoaded', () => {
    const signupForm = document.getElementById('signup-form');
    const loginForm = document.getElementById('login-form');
    const profileForm = document.getElementById('profile-form'); // For updating bodyShape, skinTone etc.

    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const name = signupForm.name.value;
            const email = signupForm.email.value;
            const password = signupForm.password.value;
            const gender = signupForm.gender.value; // Assuming a select field with name="gender"

            try {
                const data = await makeApiRequest('/auth/register', 'POST', { name, email, password, gender });
                localStorage.setItem('authToken', data.token);
                localStorage.setItem('userInfo', JSON.stringify({ // Store minimal info
                    _id: data._id,
                    name: data.name,
                    email: data.email,
                    isAdmin: data.isAdmin,  
                    bodyShape: data.bodyShape, // Initially undefined
                    skinTone: data.skinTone   // Initially undefined
                }));
                updateLoginStatusUI();
                updateCartCount();
                updateWishlistCount();
                displayMessage('signup-message', 'Registration successful! Redirecting...', false);
                window.location.href = 'index.html'; // Or profile setup page
            } catch (error) {
                displayMessage('signup-message', error.message || 'Registration failed.', true);
            }
        });
    }

    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = loginForm.email.value;
            const password = loginForm.password.value;

            try {
                const data = await makeApiRequest('/auth/login', 'POST', { email, password });
                localStorage.setItem('authToken', data.token);
                localStorage.setItem('userInfo', JSON.stringify({
                    _id: data._id,
                    name: data.name,
                    email: data.email,
                    isAdmin: data.isAdmin,
                    bodyShape: data.bodyShape,
                    skinTone: data.skinTone
                }));
                updateLoginStatusUI();
                updateCartCount();
                updateWishlistCount();
                displayMessage('login-message', 'Login successful! Redirecting...', false);
                // Redirect to previous page or homepage
                const redirectUrl = new URLSearchParams(window.location.search).get('redirect') || 'index.html';
                window.location.href = redirectUrl;
            } catch (error) {
                displayMessage('login-message', error.message || 'Login failed. Invalid credentials.', true);
            }
        });
    }

    if (profileForm) {
        // Populate form if editing existing profile
        const userInfo = getUserInfo();
        if (userInfo) {
            profileForm.name.value = userInfo.name || '';
            profileForm.email.value = userInfo.email || '';
            // Password field should typically be separate or handled with "current password"
            if (profileForm.gender) profileForm.gender.value = userInfo.gender || '';
            if (profileForm.bodyShape) profileForm.bodyShape.value = userInfo.bodyShape || '';
            if (profileForm.skinTone) profileForm.skinTone.value = userInfo.skinTone || '';
        } else {
            // If not logged in, redirect from profile page
            // window.location.href = 'login.html?redirect=profile.html';
        }


        profileForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const name = profileForm.name.value;
            const email = profileForm.email.value;
            const password = profileForm.password.value; // Optional: only if user wants to change
            const gender = profileForm.gender ? profileForm.gender.value : undefined;
            const bodyShape = profileForm.bodyShape ? profileForm.bodyShape.value : undefined;
            const skinTone = profileForm.skinTone ? profileForm.skinTone.value : undefined;

            const updateData = { name, email };
            if (password) updateData.password = password;
            if (gender) updateData.gender = gender;
            if (bodyShape) updateData.bodyShape = bodyShape;
            if (skinTone) updateData.skinTone = skinTone;

            try {
                const data = await makeApiRequest('/auth/profile', 'PUT', updateData, true);
                // Update localStorage with new info
                localStorage.setItem('userInfo', JSON.stringify({
                    _id: data._id,
                    name: data.name,
                    email: data.email,
                    isAdmin: data.isAdmin,
                    bodyShape: data.bodyShape,
                    skinTone: data.skinTone
                }));
                updateLoginStatusUI(); // Reflect name change if any
                displayMessage('profile-message', 'Profile updated successfully!', false);
            } catch (error) {
                displayMessage('profile-message', error.message || 'Profile update failed.', true);
            }
        });
    }
});

function handleLogout() { // Can be called from a logout button anywhere
    localStorage.removeItem('authToken');
    localStorage.removeItem('userInfo');
    updateLoginStatusUI();
    updateCartCount();
    updateWishlistCount();
    // Redirect to home or login page
    window.location.href = 'index.html';
}