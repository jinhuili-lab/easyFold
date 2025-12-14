import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, DateTime, UniqueConstraint,
    Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.getenv("AF3_GUI_DB", "sqlite:///./af3_gui.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("username", name="uq_username"),)

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_uid = Column(String(64), nullable=False, default="", index=True)

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(128), nullable=False, default="AF3 Job")
    status = Column(String(32), nullable=False, default="submitted")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)

    input_fasta = Column(Text, nullable=False, default="")
    params_json = Column(Text, nullable=False, default="{}")
    workdir = Column(String(512), nullable=False, default="")
    result_path = Column(String(512), nullable=False, default="")
    log_path = Column(String(512), nullable=False, default="")
    docker_cmd = Column(Text, nullable=False, default="")

    owner = relationship("User")



class ResultArtifact(Base):
    __tablename__ = "result_artifacts"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    kind = Column(String(64), nullable=False)  # summary, plddt, pae, contact, domains, compare, domain_pdb
    path = Column(String(1024), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    job = relationship("Job", backref="artifacts")


class AdminProfile(Base):
    __tablename__ = "admin_profile"
    id = Column(Integer, primary_key=True, default=1)

    input_dir = Column(String(512), nullable=False, default="")
    output_dir = Column(String(512), nullable=False, default="")
    models_dir = Column(String(512), nullable=False, default="")
    afdb_dir = Column(String(512), nullable=False, default="")

    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
