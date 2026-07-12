-- Ground-truth feature detections: one mold per part detection run, with its
-- detected features. Mirrors the conventions in 20260706000001_tools.sql
-- (uuid pk, updated_at trigger, RLS with anon/authenticated read + service_role
-- write). The `metadata` jsonb columns are lossless: they hold the full raw
-- feature-graph node / graph-level fields so a saved mold can be reconstructed
-- back into a feature_graph_cascade.json the planner accepts.
--
-- Persist a detection run with: ground_truth_store.insert_mold_with_features(...)

create table if not exists public.molds (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  step_file_ref text,
  detection_version text not null,
  metadata jsonb not null default '{}'::jsonb,   -- graph-level fields: schema_version,
                                                 -- part_id, approach_frame, edges,
                                                 -- stock_face_ids, setup_descriptor, extents
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Compare re-runs against the same baseline: a part is (name, detection_version).
create unique index if not exists molds_name_version_idx
  on public.molds (name, detection_version);

create table if not exists public.features (
  id uuid primary key default gen_random_uuid(),
  mold_id uuid not null references public.molds (id) on delete cascade,
  feature_type text not null,                    -- cascade node class_name
  face_ids jsonb not null default '[]'::jsonb,   -- sorted list[int] of B-rep face ids
  dimensions jsonb not null default '{}'::jsonb, -- normalized numeric dims pulled from params
  depth double precision,                        -- normalized depth pulled from params
  metadata jsonb not null default '{}'::jsonb,   -- full raw node (params/approach/slope_profile)
  created_at timestamptz not null default now()
);

create index if not exists features_mold_id_idx
  on public.features (mold_id);

create index if not exists features_feature_type_idx
  on public.features (feature_type);

create or replace function public.set_molds_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists molds_set_updated_at on public.molds;
create trigger molds_set_updated_at
  before update on public.molds
  for each row
  execute function public.set_molds_updated_at();

-- Atomic mold + features insert. PostgREST cannot span a transaction across
-- separate REST calls, so the "one transaction" guarantee lives here: a plpgsql
-- function runs in a single implicit transaction, so a failure inserting any
-- feature rolls back the mold too. Called from Python via client.rpc(...).
create or replace function public.insert_mold_with_features(
  p_mold jsonb,
  p_features jsonb
)
returns uuid
language plpgsql
as $$
declare
  new_mold_id uuid;
begin
  insert into public.molds (name, step_file_ref, detection_version, metadata)
  values (
    p_mold ->> 'name',
    p_mold ->> 'step_file_ref',
    p_mold ->> 'detection_version',
    coalesce(p_mold -> 'metadata', '{}'::jsonb)
  )
  returning id into new_mold_id;

  insert into public.features (mold_id, feature_type, face_ids, dimensions, depth, metadata)
  select
    new_mold_id,
    f ->> 'feature_type',
    coalesce(f -> 'face_ids', '[]'::jsonb),
    coalesce(f -> 'dimensions', '{}'::jsonb),
    nullif(f ->> 'depth', '')::double precision,
    coalesce(f -> 'metadata', '{}'::jsonb)
  from jsonb_array_elements(coalesce(p_features, '[]'::jsonb)) as f;

  return new_mold_id;
end;
$$;

alter table public.molds enable row level security;
alter table public.features enable row level security;

drop policy if exists molds_read_all on public.molds;
create policy molds_read_all
  on public.molds
  for select
  to anon, authenticated
  using (true);

drop policy if exists molds_service_write on public.molds;
create policy molds_service_write
  on public.molds
  for all
  to service_role
  using (true)
  with check (true);

drop policy if exists features_read_all on public.features;
create policy features_read_all
  on public.features
  for select
  to anon, authenticated
  using (true);

drop policy if exists features_service_write on public.features;
create policy features_service_write
  on public.features
  for all
  to service_role
  using (true)
  with check (true);
