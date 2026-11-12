-- =====================================================================
-- CERBERUS cloud setup for Supabase.
-- Paste this whole file into Supabase > SQL Editor > New query > Run.
-- Safe to run again: everything is "create if missing" / "replace".
--
-- What lives in the cloud:
--   * accounts            -> Supabase Auth (auth.users), nothing to create here
--   * incident records    -> public.incidents
--   * flag images         -> storage bucket "evidence" (private)
-- Videos never leave the PC that recorded them.
--
-- Every row and every image belongs to one account. Row Level Security
-- makes sure an account can only ever read or change its own data, even
-- though the desktop app ships with the public (anon / publishable) key.
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. Incident records
-- ---------------------------------------------------------------------
create table if not exists public.incidents (
    id             bigint generated always as identity primary key,
    owner          uuid not null default auth.uid()
                   references auth.users (id) on delete cascade,
    device_id      text not null,          -- which PC recorded it (videos stay there)
    local_id       bigint not null,        -- the incident's id in that PC's incidents.db
    session_id     text not null,
    camera_id      text,
    track_id       integer,
    violation_type text not null,
    confidence     real,
    start_offset   real,                   -- seconds into the local recording
    end_offset     real,
    recording      text,                   -- local recording path on device_id
    thumb_path     text,                   -- object name in the "evidence" bucket
    dismissed      boolean not null default false,
    created_at     timestamptz,
    synced_at      timestamptz not null default now(),
    unique (owner, device_id, local_id)
);

-- Added later: the session name the user typed (session_id may carry a
-- short suffix to keep two accounts' sessions with the same name apart).
alter table public.incidents add column if not exists session_name text;

-- Added later: when the flag was last dismissed or restored. Any PC signed
-- in to the account can dismiss/restore a flag; the newest change wins.
alter table public.incidents add column if not exists dismissed_at timestamptz;

create index if not exists incidents_owner_device_dismissed_idx
    on public.incidents (owner, device_id, dismissed_at);

-- The PC that recorded a flag keeps uploading it (for example while the
-- incident is still growing). This rule stops such an upload from undoing a
-- newer dismiss/restore made on another PC.
create or replace function public.incidents_keep_newest_dismiss()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    if old.dismissed_at is not null
       and (new.dismissed_at is null or new.dismissed_at < old.dismissed_at) then
        new.dismissed := old.dismissed;
        new.dismissed_at := old.dismissed_at;
    end if;
    return new;
end;
$$;

drop trigger if exists incidents_keep_newest_dismiss on public.incidents;
create trigger incidents_keep_newest_dismiss
    before update on public.incidents
    for each row execute function public.incidents_keep_newest_dismiss();

create index if not exists incidents_owner_session_idx
    on public.incidents (owner, session_id);

alter table public.incidents enable row level security;

drop policy if exists "incidents: owner can read"   on public.incidents;
drop policy if exists "incidents: owner can insert" on public.incidents;
drop policy if exists "incidents: owner can update" on public.incidents;
drop policy if exists "incidents: owner can delete" on public.incidents;

create policy "incidents: owner can read" on public.incidents
    for select to authenticated using (owner = (select auth.uid()));
create policy "incidents: owner can insert" on public.incidents
    for insert to authenticated with check (owner = (select auth.uid()));
create policy "incidents: owner can update" on public.incidents
    for update to authenticated
    using (owner = (select auth.uid())) with check (owner = (select auth.uid()));
create policy "incidents: owner can delete" on public.incidents
    for delete to authenticated using (owner = (select auth.uid()));


-- ---------------------------------------------------------------------
-- 2. Flag images: private bucket, one folder per account
--    Object names look like  <user id>/<device id>/<session>/<camera>/<file>.jpg
-- ---------------------------------------------------------------------
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('evidence', 'evidence', false, 1048576, array['image/jpeg'])
on conflict (id) do update
    set public = false, file_size_limit = 1048576, allowed_mime_types = array['image/jpeg'];

drop policy if exists "evidence: owner can read"   on storage.objects;
drop policy if exists "evidence: owner can upload" on storage.objects;
drop policy if exists "evidence: owner can update" on storage.objects;
drop policy if exists "evidence: owner can delete" on storage.objects;

create policy "evidence: owner can read" on storage.objects
    for select to authenticated
    using (bucket_id = 'evidence'
           and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "evidence: owner can upload" on storage.objects
    for insert to authenticated
    with check (bucket_id = 'evidence'
                and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "evidence: owner can update" on storage.objects
    for update to authenticated
    using (bucket_id = 'evidence'
           and (storage.foldername(name))[1] = (select auth.uid())::text)
    with check (bucket_id = 'evidence'
                and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "evidence: owner can delete" on storage.objects
    for delete to authenticated
    using (bucket_id = 'evidence'
           and (storage.foldername(name))[1] = (select auth.uid())::text);


-- ---------------------------------------------------------------------
-- 3. "Delete my account"
--    The app's public key cannot delete accounts, so this function does it
--    for the signed-in user only (auth.uid()). Their incident rows go with
--    it (on delete cascade). The app removes their images first.
-- ---------------------------------------------------------------------
create or replace function public.delete_my_account()
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
    if auth.uid() is null then
        raise exception 'Not signed in';
    end if;
    delete from auth.users where id = auth.uid();
end;
$$;

revoke all on function public.delete_my_account() from public, anon;
grant execute on function public.delete_my_account() to authenticated;
