const API_BASE = "http://127.0.0.1:8000";

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  return response.json();
}

// Supplier prediction
async function predictSupplier() {
  return postJson(`${API_BASE}/predict/supplier`, {
    quantity: 12,
    unit_price: 18.5,
    total_ht: 222,
    total_ttc: 265,
    governorate: "Tunis",
    city: "Tunis",
  });
}

// Sell prediction
async function predictSell() {
  const response = await fetch(`${API_BASE}/predict/sell`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

// Promote prediction
async function predictPromote() {
  const response = await fetch(`${API_BASE}/predict/promote`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

// Example usage
(async () => {
  try {
    const supplier = await predictSupplier();
    console.log("Supplier prediction:", supplier);

    const sell = await predictSell();
    console.log("Sell prediction:", sell);

    const promote = await predictPromote();
    console.log("Promote prediction:", promote);
  } catch (error) {
    console.error(error);
  }
})();
