"""create_main_tables
Revision ID: f15e4d3fc75c
Revises:
Create Date: 2023-01-30 22:15:46.925996
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic
revision = "f15e4d3fc75c"
down_revision = None
branch_labels = None
depends_on = None


def create_images_table() -> None:
    op.create_table(
        "images",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("filename", sa.Text, nullable=True),
        sa.Column("file", sa.Text, nullable=False),
        sa.Column("predictions", sa.JSON, nullable=True),
        sa.Column("ground_truth", sa.Text, nullable=True),
    )


def upgrade() -> None:
    create_images_table()


def downgrade() -> None:
    op.drop_table("images")
