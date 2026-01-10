const express = require("express");
const router = express.Router();

router.post("/predict", (req, res) => {
  res.json({ message: "Predict API is working" });
});

module.exports = router;
