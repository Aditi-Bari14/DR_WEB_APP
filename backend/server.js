const express = require("express");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json());

const predictRoutes = require("./routes/predictRoutes");

// THIS LINE MUST BE EXACT
app.use("/api", predictRoutes);

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
