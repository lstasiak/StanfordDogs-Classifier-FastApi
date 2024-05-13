from backend.app.app.models.user_model import User
from oso import Oso  # (1)


oso = Oso()  # (2)

# load policies
oso.register_class(User)

oso.load_files(["backend/app/app/core/authz.polar"])


def is_authorized(actor: User, action: str, resource, **kwargs):
    return oso.is_allowed(actor=actor, action=action, resource=resource, **kwargs)
