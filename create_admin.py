import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wildfirewatch.settings")
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

username = os.environ.get("ADMIN_USERNAME")
password = os.environ.get("ADMIN_PASSWORD")

if not username or not password:
    raise RuntimeError("ADMIN_USERNAME and ADMIN_PASSWORD must be set.")

if User.objects.filter(username=username).exists():
    print(f"Admin user '{username}' already exists.")
else:
    User.objects.create_superuser(
        username=username,
        email="",
        password=password,
    )
    print(f"Admin user '{username}' created successfully.")