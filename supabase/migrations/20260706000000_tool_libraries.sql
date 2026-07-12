-- Fusion/Toolpath tool libraries stored as jsonb payloads.
-- Seed from repo: python scripts/seed_tool_libraries.py

create table if not exists public.tool_libraries (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique,
  library_name text not null,
  display_name text not null,
  fusion_version integer,
  enabled boolean not null default true,
  content jsonb not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists tool_libraries_enabled_slug_idx
  on public.tool_libraries (enabled, slug);

create or replace function public.set_tool_libraries_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists tool_libraries_set_updated_at on public.tool_libraries;
create trigger tool_libraries_set_updated_at
  before update on public.tool_libraries
  for each row
  execute function public.set_tool_libraries_updated_at();

alter table public.tool_libraries enable row level security;

drop policy if exists tool_libraries_read_enabled on public.tool_libraries;
create policy tool_libraries_read_enabled
  on public.tool_libraries
  for select
  to anon, authenticated
  using (enabled = true);

drop policy if exists tool_libraries_service_write on public.tool_libraries;
create policy tool_libraries_service_write
  on public.tool_libraries
  for all
  to service_role
  using (true)
  with check (true);
