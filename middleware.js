/* Sign-in gate for the whole site. Runs on Vercel before any file is served.
 *
 * A visitor must sign in with Google, and the Google email must be on the
 * ALLOWED_EMAILS list (a Vercel environment variable). Nothing is sent to the
 * browser until that check passes. The list is re-checked on every request, so
 * removing an email locks that person out straight away.
 *
 * Environment variables (set in Vercel):
 *   ALLOWED_EMAILS   comma, space or newline separated list of Google emails
 *   SESSION_SECRET   long random string used to sign the login cookie
 *   GOOGLE_CLIENT_ID optional; defaults to this site's public client ID
 */
import { next } from '@vercel/functions/middleware';

const CLIENT_ID = process.env.GOOGLE_CLIENT_ID ||
  '258234360925-9s9f5cs405ccmp5919k1ic2cq8cbs204.apps.googleusercontent.com';
const COOKIE = 'gre_session';
const MAX_AGE = 7 * 24 * 3600; // seconds; sign in again after a week
const JWKS_URL = 'https://www.googleapis.com/oauth2/v3/certs';
const enc = new TextEncoder();

/* ---------- small helpers ---------- */
function b64urlFromBytes(bytes) {
  let s = '';
  bytes = new Uint8Array(bytes);
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
function bytesFromB64url(str) {
  str = str.replace(/-/g, '+').replace(/_/g, '/');
  while (str.length % 4) str += '=';
  const bin = atob(str);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function jsonFromB64url(str) {
  return JSON.parse(new TextDecoder().decode(bytesFromB64url(str)));
}
function allowedEmails() {
  return (process.env.ALLOWED_EMAILS || '')
    .split(/[\s,;]+/).map(function (e) { return e.trim().toLowerCase(); }).filter(Boolean);
}
function secret() {
  const s = process.env.SESSION_SECRET || '';
  return s.length >= 16 ? s : null;
}
function readCookie(request, name) {
  const header = request.headers.get('cookie') || '';
  const parts = header.split(/;\s*/);
  for (let i = 0; i < parts.length; i++) {
    const eq = parts[i].indexOf('=');
    if (eq > 0 && parts[i].slice(0, eq) === name) return parts[i].slice(eq + 1);
  }
  return null;
}
function html(body, status) {
  return new Response(body, {
    status: status || 200,
    headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' }
  });
}
function json(obj, status, extraHeaders) {
  const headers = { 'content-type': 'application/json', 'cache-control': 'no-store' };
  Object.keys(extraHeaders || {}).forEach(function (k) { headers[k] = extraHeaders[k]; });
  return new Response(JSON.stringify(obj), { status: status || 200, headers: headers });
}

/* ---------- the login cookie: "<payload>.<hmac>" ---------- */
async function hmacKey(usage) {
  return crypto.subtle.importKey('raw', enc.encode(secret()), { name: 'HMAC', hash: 'SHA-256' }, false, usage);
}
async function makeSession(email) {
  const payload = b64urlFromBytes(enc.encode(JSON.stringify({ e: email, x: Date.now() + MAX_AGE * 1000 })));
  const mac = await crypto.subtle.sign('HMAC', await hmacKey(['sign']), enc.encode(payload));
  return payload + '.' + b64urlFromBytes(mac);
}
async function readSession(request) {
  const value = readCookie(request, COOKIE);
  if (!value || !secret()) return null;
  const dot = value.indexOf('.');
  if (dot < 1) return null;
  const payload = value.slice(0, dot);
  try {
    const ok = await crypto.subtle.verify('HMAC', await hmacKey(['verify']), bytesFromB64url(value.slice(dot + 1)), enc.encode(payload));
    if (!ok) return null;
    const data = jsonFromB64url(payload);
    if (!data || typeof data.e !== 'string' || !(data.x > Date.now())) return null;
    return allowedEmails().indexOf(data.e.toLowerCase()) >= 0 ? data.e : null;
  } catch (e) {
    return null;
  }
}
function sessionCookie(value, maxAge) {
  return COOKIE + '=' + value + '; Path=/; Max-Age=' + maxAge + '; HttpOnly; Secure; SameSite=Lax';
}

/* ---------- checking the Google ID token ---------- */
let jwksCache = { keys: null, at: 0 };
async function googleKey(kid) {
  if (!jwksCache.keys || Date.now() - jwksCache.at > 3600 * 1000 || !jwksCache.keys.some(function (k) { return k.kid === kid; })) {
    const res = await fetch(JWKS_URL);
    if (!res.ok) throw new Error('could not load Google keys');
    jwksCache = { keys: (await res.json()).keys || [], at: Date.now() };
  }
  const jwk = jwksCache.keys.filter(function (k) { return k.kid === kid; })[0];
  if (!jwk) throw new Error('unknown signing key');
  return crypto.subtle.importKey('jwk', jwk, { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['verify']);
}
async function verifyGoogleToken(token) {
  const parts = String(token || '').split('.');
  if (parts.length !== 3) throw new Error('malformed token');
  const header = jsonFromB64url(parts[0]);
  if (header.alg !== 'RS256') throw new Error('bad algorithm');
  const key = await googleKey(header.kid);
  const valid = await crypto.subtle.verify('RSASSA-PKCS1-v1_5', key, bytesFromB64url(parts[2]), enc.encode(parts[0] + '.' + parts[1]));
  if (!valid) throw new Error('bad signature');
  const p = jsonFromB64url(parts[1]);
  if (p.iss !== 'https://accounts.google.com' && p.iss !== 'accounts.google.com') throw new Error('bad issuer');
  if (p.aud !== CLIENT_ID) throw new Error('wrong audience');
  if (!(p.exp * 1000 > Date.now() - 60000)) throw new Error('token expired');
  if (p.email_verified !== true && p.email_verified !== 'true') throw new Error('email not verified');
  if (typeof p.email !== 'string') throw new Error('no email');
  return p.email.toLowerCase();
}

async function handleLogin(request) {
  if (!secret() || !allowedEmails().length) return json({ error: 'not_configured' }, 503);
  let body;
  try { body = await request.json(); } catch (e) { return json({ error: 'bad_request' }, 400); }
  let email;
  try { email = await verifyGoogleToken(body && body.credential); } catch (e) { return json({ error: 'invalid_token' }, 401); }
  if (allowedEmails().indexOf(email) < 0) return json({ error: 'not_allowed', email: email }, 403);
  return json({ ok: true }, 200, { 'set-cookie': sessionCookie(await makeSession(email), MAX_AGE) });
}

/* ---------- the sign-in page ---------- */
function loginPage() {
  return html('<!doctype html><html lang="en"><head><meta charset="utf-8">' +
    '<meta name="viewport" content="width=device-width,initial-scale=1">' +
    '<meta name="robots" content="noindex,nofollow"><title>GRE Practice Tests</title>' +
    '<style>' +
    ':root{color-scheme:light dark}' +
    'body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:16px;box-sizing:border-box;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#e9edf1;color:#1c2630}' +
    '.box{background:#fff;border:1px solid #c5cfd9;border-radius:6px;padding:28px 24px;max-width:380px;width:100%;text-align:center}' +
    'h1{font:600 26px/1.2 Georgia,serif;margin:0 0 8px}p{margin:0 0 20px;color:#4b5a69}' +
    '#btn{display:flex;justify-content:center;min-height:44px}' +
    '#msg{margin:16px 0 0;font-size:14px;min-height:20px}#msg.err{color:#b3261e}' +
    '@media (prefers-color-scheme:dark){body{background:#12171c;color:#e6ecf2}.box{background:#1b232b;border-color:#33414e}p{color:#9fb0c0}#msg.err{color:#ff8a80}}' +
    '</style></head><body><div class="box"><h1>GRE Practice Tests</h1>' +
    '<p>This site is private. Sign in with the Google account you were given access with.</p>' +
    '<div id="btn"></div><div id="msg" role="status"></div></div>' +
    '<script src="https://accounts.google.com/gsi/client" async defer></script>' +
    '<script>' +
    'var msg=document.getElementById("msg");' +
    'function say(t,err){msg.textContent=t;msg.className=err?"err":""}' +
    'function onCredential(r){say("Checking\\u2026");' +
    'fetch("/_auth",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({credential:r.credential})})' +
    '.then(function(res){return res.json().then(function(d){return{ok:res.ok,d:d}})})' +
    '.then(function(x){if(x.ok){location.replace("/")}' +
    'else if(x.d.error==="not_allowed"){say((x.d.email||"That account")+" does not have access. Ask the owner to add it, or sign in with another Google account.",true)}' +
    'else if(x.d.error==="not_configured"){say("The site is not set up yet. Please try again later.",true)}' +
    'else{say("Sign-in failed. Please try again.",true)}})' +
    '.catch(function(){say("Could not reach the server. Please try again.",true)})}' +
    'function boot(){if(!window.google||!google.accounts){return setTimeout(boot,150)}' +
    'google.accounts.id.initialize({client_id:' + JSON.stringify(CLIENT_ID) + ',callback:onCredential,auto_select:false});' +
    'google.accounts.id.renderButton(document.getElementById("btn"),{theme:"outline",size:"large",text:"signin_with",shape:"rectangular"})}' +
    'boot();' +
    '</script></body></html>', 401);
}

/* ---------- entry point ---------- */
export default async function middleware(request) {
  const url = new URL(request.url);
  if (url.pathname === '/_auth' && request.method === 'POST') return handleLogin(request);
  if (url.pathname === '/_logout') {
    return new Response(null, { status: 302, headers: { location: '/', 'set-cookie': sessionCookie('', 0), 'cache-control': 'no-store' } });
  }
  if (!secret() || !allowedEmails().length) {
    return html('<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>GRE Practice Tests</title>' +
      '<body style="font:16px system-ui;padding:24px"><h1>GRE Practice Tests</h1><p>This site is private and is not set up for sign-in yet.</p></body>', 503);
  }
  if (await readSession(request)) return next();
  if (request.method !== 'GET' && request.method !== 'HEAD') return json({ error: 'unauthorized' }, 401);
  return loginPage();
}
