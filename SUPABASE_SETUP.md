# Connecting CERBERUS to Supabase (free)

With this set up, **accounts, incident records, and flag images** live online in your own Supabase project. **Videos stay on the PC that recorded them.** Without it, CERBERUS keeps working with local accounts on each PC.

It takes about 15 minutes. Supabase's dashboard changes from time to time, so a menu name may differ slightly from what's written here.

---

## 1. Create the project

1. Go to <https://supabase.com> and sign up (free; GitHub or email).
2. Click **New project**.
   - **Name:** `cerberus`
   - **Database password:** click *Generate*, then save it somewhere private. CERBERUS doesn't need it, but you will if you ever manage the database directly.
   - **Region:** the closest one to where the app is used (for example *Southeast Asia (Singapore)*).
   - **Plan:** Free.
3. Wait a minute or two for the project to finish setting up.

## 2. Create the tables, storage, and security rules

1. In the left menu, open **SQL Editor** and click **New query**.
2. Open [`supabase/setup.sql`](supabase/setup.sql) from this project, copy all of it, and paste it in.
3. Click **Run**. You should see *Success. No rows returned*.

This creates:
- the `incidents` table, where each account can only see its own rows,
- a private `evidence` storage bucket for flag images, with one folder per account,
- a `delete_my_account()` function, so the in-app **Delete account** button works.

It is safe to run again if you change anything. **If you set up the project before session names were added, run it once more** so flags from other PCs show their session name and can be dismissed or restored from any PC.

## 3. Set up sign-in

In the left menu, open **Authentication**.

### Password rules
Under **Sign In / Providers → Email** (sometimes listed under *Policies*):
- Keep **Email** enabled.
- Set **Minimum password length** to `8`.
- Set **Password requirements** to *lowercase, uppercase, digits and symbols*, so it matches the app.

### MFA
Under **Multi-Factor** (or *MFA*), make sure **TOTP / Authenticator app** is enabled. It is on by default.

### Email confirmation and password-reset emails

CERBERUS uses **codes from email** (6 to 10 digits, set by *Email OTP length*) (not links), because a desktop app can't open a link sent to a browser.

**Important free-tier limit:** Supabase's built-in email sender only delivers to the email addresses of people on your Supabase team, and only a few emails per hour. For anyone else to get sign-up and password-reset codes, you need your own email sender (SMTP). See step 3b.

**3a. Quick start (testing):** under **Sign In / Providers → Email**, turn **Confirm email** OFF. New accounts work immediately with no email. Password reset will still only reach team members' addresses until you do 3b.

**3b. For real use:** set up a free SMTP sender, then turn **Confirm email** back ON.
1. Pick a sender, for example:
   - **Gmail**: turn on 2-Step Verification on the Google account, then create an *App password* (Google Account → Security → App passwords). Host `smtp.gmail.com`, port `587`, username = the Gmail address, password = the app password.
   - **Brevo** (free: 300 emails/day): sign up, then copy the SMTP host, port, login, and key from its *SMTP & API* page.
2. In Supabase: **Authentication → Emails → SMTP Settings** (or *Project Settings → Authentication → SMTP*). Turn on **Custom SMTP** and fill in the sender email, name (`CERBERUS`), host, port, username, and password.
3. Change two email templates so they send the code. Go to **Authentication → Emails → Templates**:
   - **Confirm signup**, message body:
     ```html
     <h2>Confirm your CERBERUS account</h2>
     <p>Your code is: <strong>{{ .Token }}</strong></p>
     <p>Enter it in the CERBERUS app. It expires in one hour.</p>
     ```
   - **Reset Password**, message body:
     ```html
     <h2>Reset your CERBERUS password</h2>
     <p>Your code is: <strong>{{ .Token }}</strong></p>
     <p>Enter it in the CERBERUS app with your new password. If you didn't ask for this, ignore this email.</p>
     ```
   Save each one.

## 4. Copy the project URL and public key

1. Open **Project Settings → API Keys** (older dashboards: *Project Settings → API*).
2. Copy:
   - the **Project URL**, for example `https://abcdxyz.supabase.co` (also shown under *Data API*),
   - the **publishable key** (`sb_publishable_...`), or on older projects the **anon public** key (a long `eyJ...` string).

> **Never use the `secret` / `service_role` key.** It bypasses every security rule, and anything put inside the app can be read by anyone who has a copy. CERBERUS refuses to start cloud mode with it.

## 5. Point CERBERUS at the project

1. In the project folder, copy `cloud_config.example.json` to **`cloud_config.json`**.
2. Fill it in:
   ```json
   {
     "supabase_url": "https://abcdxyz.supabase.co",
     "supabase_key": "sb_publishable_..."
   }
   ```
   `cloud_config.json` is in `.gitignore`, so it is not pushed to GitHub.
3. Start the app:
   ```bash
   python desktop.py
   ```
   The console should print `accounts: cloud (Supabase) https://abcdxyz.supabase.co`, and the sign-in screen asks for an **email** instead of a username.
4. Run `build_desktop.bat`. The build includes `cloud_config.json`, so every installed copy uses your project.

## 6. Check it works

1. In the app, **Sign up** with an email and password. In Supabase, **Authentication → Users** now lists that email.
2. Start a session, start a camera, and trigger a flag (for example, hold up a phone).
3. Within about 10 seconds, the Sessions page shows **Cloud: up to date**. In Supabase:
   - **Table Editor → incidents** has the flag,
   - **Storage → evidence** has the image, under a folder named after your user id.
4. Delete that test session in the app. Within about 10 seconds its row and image disappear from Supabase.

---

## How it behaves

| Situation | What happens |
|---|---|
| No internet while monitoring | Detection and recording keep working. Flags queue locally and upload when the connection comes back. The Sessions page shows *Cloud: offline, N waiting*. |
| No internet at sign-in | Sign-in fails with a "can't reach the cloud" message. Cloud accounts need internet to sign in. |
| Same account on two PCs | Both PCs upload to the same account. Each flag records which PC (`device_id`) holds its video. |
| Delete a session | It is removed from this PC and its rows and images are removed from the cloud. If offline, the cloud part happens on the next sync. |
| Delete account | All of the account's cloud images and rows (from every PC), the account itself, and its sessions on this PC are removed. |
| Old local test accounts | Never uploaded. Their sessions stay on this PC, and they can't sign in while cloud mode is on. |
| Lost authenticator app | There are no recovery codes for cloud accounts. In Supabase, **Authentication → Users → (user) → remove the MFA factor**, then the user can sign in with just their password. |

## Free plan limits

- **Pauses after about 7 days with no activity.** Nothing is lost: open the project in the dashboard and click **Restore**. Using the app at least once a week avoids it.
- **500 MB database, 1 GB file storage.** A flag image is about 20–40 KB, so 1 GB holds roughly 30,000 flags.
- **Free projects per account:** 2.

Check <https://supabase.com/pricing> for the current numbers.
