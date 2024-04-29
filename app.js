// Import all required modeles
const express = require("express");
const Wallet = require("./wallet");
const bodyParser = require("body-parser");
const TransactionPool = require("./transaction-pool");
const P2pserver = require("./p2p-server");
const Validators = require("./validators");
const Blockchain = require("./blockchain");
const BlockPool = require("./block-pool");
const CommitPool = require("./commit-pool");
const PreparePool = require("./prepare-pool");
const MessagePool = require("./message-pool");
const { NUMBER_OF_NODES } = require("./config");
const HTTP_PORT = process.env.HTTP_PORT || 3000;
const multer = require('multer');
const path = require('path');

// Instantiate all objects
const app = express();
app.use(bodyParser.json());

// Configure multer for handling file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/'); // The uploaded files will be saved in the 'uploads/' directory
  },
  filename: function (req, file, cb) {
    cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

app.use(express.static('public')); // Serve the HTML file

const wallet = new Wallet(process.env.SECRET);
const transactionPool = new TransactionPool();
const validators = new Validators(NUMBER_OF_NODES);
const blockchain = new Blockchain(validators);
const blockPool = new BlockPool();
const preparePool = new PreparePool();
const commitPool = new CommitPool();
const messagePool = new MessagePool();
const p2pserver = new P2pserver(
  blockchain,
  transactionPool,
  wallet,
  blockPool,
  preparePool,
  commitPool,
  messagePool,
  validators
);

// sends all transactions in the transaction pool to the user
app.get("/transactions", (req, res) => {
  res.json(transactionPool.transactions);
});

// sends the entire chain to the user
app.get("/blocks", (req, res) => {
  res.json(blockchain.chain);
});

// creates transactions for the sent data
// app.post("/transact", (req, res) => {
//   const { data } = req.body;
//   const transaction = wallet.createTransaction(data);
//   transactionPool.addTransaction(transaction);
//   p2pserver.broadcastTransaction(transaction);
//   res.redirect("/transactions");
// });

app.post("/transact", upload.single('file'), (req, res) => {
  const file = req.file; // This will contain the file information
  if (!file) {
    return res.status(400).send('No file uploaded.');
  }

  // Use the file information (e.g., file.path) to process the file as needed
  const transaction = wallet.createTransaction(file.path);
  transactionPool.addTransaction(transaction);
  p2pserver.broadcastTransaction(transaction);
  res.redirect("/transactions");
});


// starts the app server
app.listen(HTTP_PORT, () => {
  console.log(`Listening on port ${HTTP_PORT}`);
});

// starts the p2p server
p2pserver.listen();


// SECRET=NODE0 P2P_PORT=5000 HTTP_PORT=3000 node app
// SECRET=NODE1 P2P_PORT=5001 HTTP_PORT=3001 PEERS=ws://localhost:5000 node app
// SECRET=NODE2 P2P_PORT=5002 HTTP_PORT=3002 PEERS=ws://localhost:5001,ws://localhost:5000 node app
// SECRET=NODE3 P2P_PORT=5003 HTTP_PORT=3003 PEERS=ws://localhost:5002,ws://localhost:5001,ws://localhost:5000 node app
// SECRET=NODE4 P2P_PORT=5004 HTTP_PORT=3004 PEERS=ws://localhost:5003,ws://localhost:5002,ws://localhost:5001,ws://localhost:5000 node app
// SECRET=NODE5 P2P_PORT=5005 HTTP_PORT=3005 PEERS=ws://localhost:5004,ws://localhost:5003,ws://localhost:5002,ws://localhost:5001,ws://localhost:5000 node app
// SECRET=NODE6 P2P_PORT=5006 HTTP_PORT=3006 PEERS=ws://localhost:5005,ws://localhost:5004,ws://localhost:5003,ws://localhost:5002,ws://localhost:5001,ws://localhost:5000 node app
// SECRET=NODE7 P2P_PORT=5007 HTTP_PORT=3007 PEERS=ws://localhost:5006,ws://localhost:5005,ws://localhost:5004,ws://localhost:5003,ws://localhost:5002,ws://localhost:5001,ws://localhost:5000 node app
// SECRET=NODE8 P2P_PORT=5008 HTTP_PORT=3008 PEERS=ws://localhost:5007,ws://localhost:5006,ws://localhost:5005,ws://localhost:5004,ws://localhost:5003,ws://localhost:5002,ws://localhost:5001,ws://localhost:5000 node app
// SECRET=NODE9 P2P_PORT=5009 HTTP_PORT=3009 PEERS=ws://localhost:5008,ws://localhost:5007,ws://localhost:5006,ws://localhost:5005,ws://localhost:5004,ws://localhost:5003,ws://localhost:5002,ws://localhost:5001,ws://localhost:5000 node app
// http://localhost:3000/transactions

