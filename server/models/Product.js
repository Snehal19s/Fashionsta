// models/Product.js
const mongoose = require('mongoose');

const ProductSchema = new mongoose.Schema({
  name:        { type: String, required: true },
  category:    { type: String, required: true, enum: ['Men', 'Women', 'Fashion Accessories', 'Sale', 'New Arrivals', 'Men Footwear', 'Women Footwear'] },
  price:       { type: Number, required: true },
  images:      [{ type: String, required: true }],
  description: { type: String, required: true },
  gender:      { type: String, required: true, enum: ["Men","Women","Unisex"] },
  bodyShapeTags:[{ type: String }],   // e.g. ["Pear","Hourglass"]
  skinToneTags:[{ type: String }],    // e.g. ["Light","Medium"]
  sizes:       [{ type: String, required:true }],    // e.g. ["S","M","L"]
  stock:       { type: Number, default: 0 },
  ratings:     { type: Number, default: 0 },
  numOfReviews:{ type: Number, default: 0 },

  // ← NEW tag fields for your recommendations:
  colorTags:   [{ type: String }],    // e.g. ["Coral","Lavender","Olive"]
  styleTags:   [{ type: String }],    // e.g. ["Bohemian","Classic","Chic"]
  fitTags:     [{ type: String }],    // e.g. ["Slim Fit","Relaxed Fit","Tailored"]
}, { timestamps: true });

module.exports = mongoose.model('Product', ProductSchema);
