// client/public/js/admin.js

document.addEventListener('DOMContentLoaded', () => {
    const adminContentArea = document.getElementById('admin-content-area');
    const userInfo = getUserInfo();

    // Protect admin routes client-side
    if (!userInfo || !userInfo.isAdmin) {
        alert('Access Denied. You are not an administrator.');
        window.location.href = 'index.html';
        return;
    }

    // Navigation listeners
    document.getElementById('admin-products-link')?.addEventListener('click', e => { e.preventDefault(); loadAdminProductsView(); });
    document.getElementById('admin-add-product-link')?.addEventListener('click', e => { e.preventDefault(); loadAddProductForm(); });
    document.getElementById('admin-orders-link')?.addEventListener('click', e => { e.preventDefault(); loadAdminOrdersView(); });
    document.getElementById('admin-users-link')?.addEventListener('click', e => { e.preventDefault(); loadAdminUsersView(); });

    // Default view
    loadAdminProductsView();

    // --- Product Management View ---
 async function loadAdminProductsView() {
        adminContentArea.innerHTML = `
            <h2>Manage Products
                <button id="show-add-product-form" class="btn btn-sm btn-success" style="float:right;">Add New Product</button>
            </h2>
            <div id="admin-product-list" class="loader"></div>
        `;
        document.getElementById('show-add-product-form').addEventListener('click', loadAddProductForm);
        const productListEl = document.getElementById('admin-product-list');
        try {
            // Use public products endpoint to list for admin
            const data = await makeApiRequest('/products?pageSize=100', 'GET', null, true);
            productListEl.classList.remove('loader');
            productListEl.innerHTML = `
                <table class="admin-table">
                    <thead>
                        <tr>
                            <th>Image</th><th>Name</th><th>Category</th><th>Price</th><th>Stock</th>
                            <th>Color Tags</th><th>Style Tags</th><th>Fit Tags</th><th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.products.map(product => `
                            <tr data-product-id="${product._id}">
                                <td><img src="http://localhost:5001${product.images[0]}" alt="${product.name}" style="width:50px;height:auto;"></td>
                                <td>${product.name}</td>
                                <td>${product.category}</td>
                                <td>₹${product.price.toFixed(2)}</td>
                                <td>${product.stock}</td>
                                <td>${product.colorTags?.join(', ')}</td>
                                <td>${product.styleTags?.join(', ')}</td>
                                <td>${product.fitTags?.join(', ')}</td>
                                <td>
                                    <button class="btn btn-sm btn-info edit-product-btn">Edit</button>
                                    <button class="btn btn-sm btn-danger delete-product-btn">Delete</button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            // Attach edit/delete handlers
            productListEl.querySelectorAll('.edit-product-btn').forEach(btn => {
                btn.addEventListener('click', e => {
                    const id = e.target.closest('tr').dataset.productId;
                    loadEditProductForm(id);
                });
            });
            productListEl.querySelectorAll('.delete-product-btn').forEach(btn => {
                btn.addEventListener('click', async e => {
                    const id = e.target.closest('tr').dataset.productId;
                    if (confirm('Are you sure you want to delete this product?')) {
                        await deleteEntity('products', id, loadAdminProductsView);
                    }
                });
            });
        } catch (error) {
            productListEl.innerHTML = `<p class="error-message">Error loading products: ${error.message}</p>`;
        }
    }
    


    function loadAddProductForm() {
        adminContentArea.innerHTML = `
            <h2>Add New Product</h2>
            <form id="add-product-form" class="form-container admin-form">
                <div class="form-group"><label>Name:</label><input type="text" name="name" required></div>
                <div class="form-group"><label>Description:</label><textarea name="description" required></textarea></div>
                <div class="form-group"><label>Price:</label><input type="number" name="price" step="0.01" required></div>
                <div class="form-group"><label>Category:</label>
                    <select name="category" required>
                        <option>Men</option><option>Women</option><option>Fashion Accessories</option><option>Sale</option><option>New Arrivals</option><option>Men Footwear</option><option>Women Footwear</option>
                    </select>
                </div>
                <div class="form-group"><label>Gender:</label>
                    <select name="gender" required>
                        <option>Men</option><option>Women</option>
                    </select>
                </div>
                <div class="form-group"><label>Stock:</label><input type="number" name="stock" required></div>
                <div class="form-group"><label>Sizes (comma-separated):</label><input type="text" name="sizes"></div>
                <div class="form-group"><label>Body Shape Tags:</label><input type="text" name="bodyShapeTags"></div>
                <div class="form-group"><label>Skin Tone Tags:</label><input type="text" name="skinToneTags"></div>
                <div class="form-group"><label>Color Tags:</label><input type="text" name="colorTags"></div>
                <div class="form-group"><label>Style Tags:</label><input type="text" name="styleTags"></div>
                <div class="form-group"><label>Fit Tags:</label><input type="text" name="fitTags"></div>
                <div class="form-group"><label>Images:</label><input type="file" name="images" multiple accept="image/*"></div>
                <button type="submit" class="btn btn-primary">Add Product</button>  
                <div id="add-product-message" class="form-message"></div>
            </form>
        `;
        document.getElementById('add-product-form').addEventListener('submit', handleProductFormSubmit);
    }

async function loadEditProductForm(productId) {
    try {
        // 🔑 Use the public GET endpoint, not /admin/products
        const product = await makeApiRequest(`/products/${productId}`, 'GET', null, true);

        adminContentArea.innerHTML = `
            <h2>Edit Product: ${product.name}</h2>
            <form id="edit-product-form" class="form-container admin-form" data-product-id="${productId}">
                <div class="form-group"><label>Name:</label>
                    <input type="text" name="name" value="${product.name}" required>
                </div>
                <div class="form-group"><label>Description:</label>
                    <textarea name="description" required>${product.description}</textarea>
                </div>
                <div class="form-group"><label>Price:</label>
                    <input type="number" name="price" value="${product.price}" step="0.01" required>
                </div>
                <div class="form-group"><label>Category:</label>
                    <select name="category" required>
                        <option value="Men" ${product.category==='Men'?'selected':''}>Men</option>
                        <option value="Women" ${product.category==='Women'?'selected':''}>Women</option>
                        <option value="Fashion Accessories" ${product.category==='Fashion Accessories'?'selected':''}>Fashion Accessories</option>
                         <option value="Sale" ${product.category==='Sale'?'selected':''}>Sale</option>
                            <option value="New Arrivals" ${product.category==='New Arrivals'?'selected':''}>New Arrivals</option>
                           <option value="Men Footwear" ${product.category==='Men Footwear'?'selected':''}>Men Footwear</option>
                            <option value="Women Footwear" ${product.category==='Women Footwear'?'selected':''}>Women Footwear</option>
                    </select>
                </div>
                <div class="form-group"><label>Gender:</label>
                    <select name="gender" required>
                        <option value="Men" ${product.gender==='Men'?'selected':''}>Men</option>
                        <option value="Women" ${product.gender==='Women'?'selected':''}>Women</option>
                        <option value="Unisex" ${product.gender==='Unisex'?'selected':''}>Unisex</option>
                    </select>
                </div>
                <div class="form-group"><label>Stock:</label>
                    <input type="number" name="stock" value="${product.stock}" required>
                </div>
                <div class="form-group"><label>Sizes (comma-separated):</label>
                    <input type="text" name="sizes" value="${product.sizes.join(',')}">
                </div>
                <div class="form-group"><label>Body Shape Tags:</label>
                    <input type="text" name="bodyShapeTags" value="${product.bodyShapeTags.join(',')}">
                </div>
                <div class="form-group"><label>Skin Tone Tags:</label>
                    <input type="text" name="skinToneTags" value="${product.skinToneTags.join(',')}">
                </div>

                <!-- NEW tag fields: -->
                <div class="form-group"><label>Color Tags:</label>
                    <input type="text" name="colorTags" value="${product.colorTags.join(',')}">
                </div>
                <div class="form-group"><label>Style Tags:</label>
                    <input type="text" name="styleTags" value="${product.styleTags.join(',')}">
                </div>
                <div class="form-group"><label>Fit Tags:</label>
                    <input type="text" name="fitTags" value="${product.fitTags.join(',')}">
                </div>

                <div class="form-group">
                    <label>Current Images:</label>
                    <div>
                      ${product.images.map(img => `<img src="http://localhost:5001${img}" style="width:50px; margin-right:5px;">`).join('')}
                    </div>
                    <label for="new_images">Upload New Images:</label>
                    <input type="file" id="new_images" name="images" multiple accept="image/*">
                </div>

                <button type="submit" class="btn btn-primary">Save Changes</button>
                <button type="button" class="btn btn-secondary" onclick="loadAdminProductsView()">Cancel</button>
                <div id="edit-product-message" class="form-message"></div>
            </form>
        `;

        document.getElementById('edit-product-form')
                .addEventListener('submit', handleProductFormSubmit);

    } catch (error) {
        adminContentArea.innerHTML = `<p class="error-message">Error loading product for editing: ${error.message}</p>`;
    }
}


    async function handleProductFormSubmit(e) {
        e.preventDefault();
        const form = e.target;
        const productId = form.dataset.productId;
        const messageEl = productId ? 'edit-product-message' : 'add-product-message';
        const formData = new FormData(form);
        const endpoint = productId ? `/admin/products/${productId}` : '/admin/products';
        const method = productId ? 'PUT' : 'POST';
        try {
            await makeApiRequest(endpoint, method, formData, true);
            displayMessage(messageEl, `Product ${productId?'updated':'added'} successfully!`, false);
            setTimeout(loadAdminProductsView, 1500);
        } catch (err) {
            displayMessage(messageEl, `Error: ${err.message}`, true);
        }
    }


    // --- User Management View ---
    async function loadAdminUsersView() {
        adminContentArea.innerHTML = '<h2>Manage Users</h2><div id="admin-user-list" class="loader"></div>';
        const userListEl = document.getElementById('admin-user-list');
        try {
            const users = await makeApiRequest('/admin/users', 'GET', null, true);
            userListEl.classList.remove('loader');
            userListEl.innerHTML = `
                <table class="admin-table">
                    <thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Admin</th><th>Actions</th></tr></thead>
                    <tbody>
                        ${users.map(user => `
                            <tr data-user-id="${user._id}">
                                <td>${user._id}</td>
                                <td>${user.name}</td>
                                <td>${user.email}</td>
                                <td>${user.isAdmin ? 'Yes' : 'No'}</td>
                                <td>
                                    ${!user.isAdmin ? `<button class="btn btn-sm btn-danger delete-user-btn">Delete</button>` : '(Admin)'}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
             userListEl.querySelectorAll('.delete-user-btn').forEach(btn => btn.addEventListener('click', async (e) => {
                const userId = e.target.closest('tr').dataset.userId;
                if (confirm('Are you sure you want to delete this user? This is irreversible.')) {
                    await deleteEntity('users', userId, loadAdminUsersView);
                }
            }));
        } catch (error) {
            userListEl.innerHTML = `<p class="error-message">Error loading users: ${error.message}</p>`;
        }
    }

    // --- Order Management View ---
    async function loadAdminOrdersView() {
        adminContentArea.innerHTML = '<h2>Manage Orders</h2><div id="admin-order-list" class="loader"></div>';
        const orderListEl = document.getElementById('admin-order-list');
        try {
            const orders = await makeApiRequest('/admin/orders', 'GET', null, true);
            orderListEl.classList.remove('loader');
            orderListEl.innerHTML = `
                <table class="admin-table">
                    <thead><tr><th>ID</th><th>User</th><th>Date</th><th>Total</th><th>Payment</th><th>Status</th><th>Actions</th></tr></thead>
                    <tbody>
                        ${orders.map(order => `
                            <tr data-order-id="${order._id}">
                                <td>${order._id}</td>
                                <td>${order.userId ? order.userId.name : 'N/A'} (${order.userId ? order.userId.email : 'N/A'})</td>
                                <td>${new Date(order.date).toLocaleDateString()}</td>
                                <td>$${order.totalPrice.toFixed(2)}</td>
                                <td>${order.paymentStatus}</td>
                                <td>
                                    <select class="order-status-select" data-order-id="${order._id}">
                                        <option value="Processing" ${order.orderStatus === 'Processing' ? 'selected' : ''}>Processing</option>
                                        <option value="Shipped" ${order.orderStatus === 'Shipped' ? 'selected' : ''}>Shipped</option>
                                        <option value="Delivered" ${order.orderStatus === 'Delivered' ? 'selected' : ''}>Delivered</option>
                                        <option value="Cancelled" ${order.orderStatus === 'Cancelled' ? 'selected' : ''}>Cancelled</option>
                                    </select>
                                </td>
                                <td><button class="btn btn-sm btn-info view-order-details-btn">Details</button></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            orderListEl.querySelectorAll('.order-status-select').forEach(select => {
                select.addEventListener('change', async (e) => {
                    const orderId = e.target.dataset.orderId;
                    const newStatus = e.target.value;
                    try {
                        await makeApiRequest(`/admin/orders/${orderId}/status`, 'PUT', { orderStatus: newStatus }, true);
                        alert('Order status updated!');
                        // Optionally refresh or just visually confirm
                    } catch (error) {
                        alert(`Failed to update status: ${error.message}`);
                        loadAdminOrdersView(); // Revert on error
                    }
                });
            });
             orderListEl.querySelectorAll('.view-order-details-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const orderId = e.target.closest('tr').dataset.orderId;
                    viewOrderDetailsModal(orderId, orders.find(o => o._id === orderId));
                });
            });
        } catch (error) {
            orderListEl.innerHTML = `<p class="error-message">Error loading orders: ${error.message}</p>`;
        }
    }

    function viewOrderDetailsModal(orderId, order) { // Simple alert for now, ideally a modal
        let details = `Order ID: ${orderId}\n`;
        details += `User: ${order.userId.name} (${order.userId.email})\n`;
        details += `Date: ${new Date(order.date).toLocaleString()}\n`;
        details += `Total: $${order.totalPrice.toFixed(2)}\n`;
        details += `Status: ${order.orderStatus}\n`;
        details += `Payment: ${order.paymentStatus}\n\n`;
        details += `Shipping To:\n${order.shippingAddress.address}\n${order.shippingAddress.city}, ${order.shippingAddress.postalCode}\n${order.shippingAddress.country}\n\n`;
        details += "Products:\n";
        order.products.forEach(p => {
            details += `- ${p.name} (Qty: ${p.quantity}, Size: ${p.size}, Price: $${p.price.toFixed(2)})\n`;
        });
        alert(details); // Replace with a proper modal display
    }

    // --- Generic Delete Function ---
    async function deleteEntity(entityType, entityId, refreshCallback) {
        // entityType should be 'products', 'users' etc. to match API path
        try {
            await makeApiRequest(`/admin/${entityType}/${entityId}`, 'DELETE', null, true);
            alert(`${entityType.slice(0, -1)} deleted successfully.`);
            refreshCallback();
        } catch (error) {
            alert(`Error deleting ${entityType.slice(0,-1)}: ${error.message}`);
        }
    }

});