-- Normalized Fusion tools (one row per tool; presets as jsonb).
-- Ingest once from local exports: python scripts/ingest_tool_libraries.py /path/to/libs

create table if not exists public.tools (
  id uuid primary key default gen_random_uuid(),
  guid text not null unique,
  tool_id text not null,
  name text,
  tool_type text not null,
  raw_type text,
  diameter_mm double precision not null,
  flute_length_mm double precision,
  max_depth_mm double precision,
  flute_count integer,
  corner_radius_mm double precision,
  point_angle_deg double precision,
  tool_material text,
  vendor text,
  source_library text not null,
  source_unit text,
  presets jsonb not null default '[]'::jsonb,
  raw jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists tools_tool_type_idx
  on public.tools (tool_type);

create index if not exists tools_source_library_idx
  on public.tools (source_library);

create or replace function public.set_tools_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists tools_set_updated_at on public.tools;
create trigger tools_set_updated_at
  before update on public.tools
  for each row
  execute function public.set_tools_updated_at();

alter table public.tools enable row level security;

drop policy if exists tools_read_all on public.tools;
create policy tools_read_all
  on public.tools
  for select
  to anon, authenticated
  using (true);

drop policy if exists tools_service_write on public.tools;
create policy tools_service_write
  on public.tools
  for all
  to service_role
  using (true)
  with check (true);
