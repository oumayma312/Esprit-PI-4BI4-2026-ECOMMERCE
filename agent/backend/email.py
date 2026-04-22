"""Email service using Resend for sending order notifications."""
import os
import logging

logger = logging.getLogger(__name__)

try:
    import resend
except ImportError:
    resend = None


class EmailService:
    """Send emails via Resend API with fallback to console logging."""

    def __init__(self):
        self.enabled = False
        self.api_key = os.getenv("RESEND_API_KEY", "")
        self.from_email = os.getenv("EMAIL_FROM", "orders@atelier-materials.com")

        if resend and self.api_key:
            resend.api_key = self.api_key
            self.enabled = True
        else:
            if not resend:
                logger.warning("Resend SDK not installed. Email sending disabled. Install with: uv pip install resend")
            elif not self.api_key:
                logger.warning("RESEND_API_KEY not set. Email sending disabled. Add to .env: RESEND_API_KEY=re_your_key")

    def send_order_notification(self, to: str, material_name: str, quantity: int,
                                unit: str, order_id: int, notes: str = "") -> dict | None:
        """Send order confirmation email to manufacturer."""
        if not self.enabled:
            # Fallback: log to console
            logger.info(
                f"[EMAIL NOT SENT] Would send to {to}:\n"
                f"  Order #{order_id}: {material_name} x{quantity} {unit}\n"
                f"  Notes: {notes}"
            )
            return None

        subject = f"Order #{order_id} - {material_name} x{quantity}"
        html = f"""
        <html>
        <body style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto; padding: 2rem;">
            <h2 style="color: #1B4332; border-bottom: 2px solid #E07A5F; padding-bottom: 0.5rem;">
                New Order Confirmed
            </h2>
            <p>Dear Supplier,</p>
            <p>An order has been confirmed and requires your attention:</p>
            <table style="width: 100%; border-collapse: collapse; margin: 1.5rem 0;">
                <tr><td style="padding: 0.5rem; border: 1px solid #ddd; font-weight: bold;">Order ID</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">#{order_id}</td></tr>
                <tr><td style="padding: 0.5rem; border: 1px solid #ddd; font-weight: bold;">Material</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">{material_name}</td></tr>
                <tr><td style="padding: 0.5rem; border: 1px solid #ddd; font-weight: bold;">Quantity</td>
                    <td style="padding: 0.5rem; border: 1px solid #ddd;">{quantity} {unit}</td></tr>
                {'<tr><td style="padding: 0.5rem; border: 1px solid #ddd; font-weight: bold;">Notes</td>'
                 f'<td style="padding: 0.5rem; border: 1px solid #ddd;">{notes}</td></tr>' if notes else ''}
            </table>
            <p style="color: #888; font-size: 0.9rem;">This is an automated notification from Atelier Materials.</p>
        </body>
        </html>
        """

        text = f"Order #{order_id}: {material_name} x{quantity} {unit}" + (f"\nNotes: {notes}" if notes else "")

        try:
            result = resend.Emails.send({
                "from": self.from_email,
                "to": [to],
                "subject": subject,
                "html": html,
                "text": text,
            })
            logger.info(f"Email sent to {to}: {result.get('id')}")
            return result
        except Exception as e:
            logger.error(f"Failed to send email to {to}: {e}")
            # Fallback to console
            logger.info(f"[EMAIL FAILED] Would send to {to}: {subject}")
            return None

    def send_notification(self, to: str, subject: str, body: str) -> dict | None:
        """Generic email sending."""
        if not self.enabled:
            logger.info(f"[EMAIL NOT SENT] to={to} subject={subject}\n{body}")
            return None

        try:
            result = resend.Emails.send({
                "from": self.from_email,
                "to": [to],
                "subject": subject,
                "html": f"<h2>{subject}</h2><p>{body}</p>",
                "text": body,
            })
            logger.info(f"Email sent to {to}: {result.get('id')}")
            return result
        except Exception as e:
            logger.error(f"Failed to send email to {to}: {e}")
            return None


# Global instance
_email_service = None


def get_email_service() -> EmailService:
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
