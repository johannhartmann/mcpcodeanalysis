# Domain Analysis Example

This example shows how the domain-driven analysis transforms traditional code understanding.

## Sample Code

```python
# File: src/billing/invoice.py

class Invoice:
    def __init__(self, customer_id, items):
        self.customer_id = customer_id
        self.items = items
        self.status = "draft"
        self.total = 0
        self.tax_rate = 0.1

    def calculate_total(self):
        subtotal = sum(item.price * item.quantity for item in self.items)
        tax = subtotal * self.tax_rate
        self.total = subtotal + tax

    def send_to_customer(self):
        if self.status != "finalized":
            raise ValueError("Cannot send draft invoice")
        # Send email logic here
        self.status = "sent"

    def mark_as_paid(self, payment_ref):
        if self.status != "sent":
            raise ValueError("Can only mark sent invoices as paid")
        self.status = "paid"
        self.payment_reference = payment_ref
```

## Traditional Analysis Output

```json
{
  "type": "class",
  "name": "Invoice",
  "methods": ["__init__", "calculate_total", "send_to_customer", "mark_as_paid"],
  "attributes": ["customer_id", "items", "status", "total", "tax_rate"],
  "line_count": 24
}
```

## Domain-Driven Analysis Output

```json
{
  "domain_entity": {
    "name": "Invoice",
    "type": "aggregate_root",
    "description": "Represents a billable document sent to customers for payment",
    "business_rules": [
      "Invoice total must include applicable taxes",
      "Only finalized invoices can be sent to customers",
      "Payment can only be recorded for sent invoices"
    ],
    "invariants": [
      "Invoice status must follow progression: draft -> finalized -> sent -> paid",
      "Total amount cannot be negative",
      "Invoice must have at least one line item"
    ],
    "responsibilities": [
      "Calculate total amount including taxes",
      "Manage invoice lifecycle states",
      "Track payment information"
    ],
    "ubiquitous_language": {
      "finalized": "Invoice is complete and ready to send",
      "sent": "Invoice has been delivered to customer",
      "paid": "Payment has been received and recorded"
    }
  },
  "bounded_context": {
    "name": "Billing",
    "description": "Handles invoice generation, tax calculation, and payment tracking"
  },
  "relationships": [
    {
      "source": "Invoice",
      "target": "Customer",
      "type": "references",
      "description": "Invoice is issued to a specific customer"
    },
    {
      "source": "Invoice",
      "target": "InvoiceItem",
      "type": "aggregates",
      "description": "Invoice contains multiple line items"
    }
  ],
  "refactoring_suggestions": [
    {
      "type": "missing_value_object",
      "severity": "medium",
      "message": "Tax calculation could be extracted to a TaxCalculator value object",
      "suggestion": "Create a Money value object to handle currency calculations"
    },
    {
      "type": "state_pattern_candidate",
      "severity": "low",
      "message": "Status transitions could benefit from State pattern",
      "suggestion": "Consider implementing InvoiceState hierarchy for cleaner state management"
    }
  ]
}
```

## Domain-Enhanced Search Examples

### Query: "Find payment processing"

**Traditional Search Results:**
- `mark_as_paid()` method in Invoice class
- `process_payment()` in PaymentGateway class

**Domain-Enhanced Results:**
- **Invoice aggregate** (Billing context) - Manages payment recording
- **Payment aggregate** (Payments context) - Handles payment transactions
- **Order aggregate** (Sales context) - Triggers payment on confirmation
- Related concepts: Refund, PaymentMethod, Transaction

### Query: "Show me the billing system"

**Domain Response:**
```
Bounded Context: Billing
- Core Purpose: Manage financial transactions and customer billing
- Aggregates: Invoice (root), CreditNote
- Entities: InvoiceItem, TaxRule
- Value Objects: Money, TaxRate, BillingPeriod
- Domain Services: TaxCalculationService, InvoiceNumberGenerator
- Integration Points:
  - Customer (from Sales context)
  - Payment (from Payments context)
```

## Benefits Demonstrated

1. **Business Understanding**: The analysis reveals the business purpose, not just code structure
2. **Hidden Rules**: Extracts implicit business rules from validation logic
3. **Context Boundaries**: Identifies that Invoice belongs to Billing, not Sales
4. **Improvement Opportunities**: Suggests domain-appropriate refactoring
5. **Richer Search**: Finds conceptually related code, not just keyword matches
