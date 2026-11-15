const $ = id => document.getElementById(id);
const states = {queued:'В очереди',running:'Скачивается',pausing:'Приостанавливается',paused:'На паузе',cancelling:'Отменяется',cancelled:'Отменено',done:'Завершено',failed:'Ошибка'};
let selected = null;
let refreshGeneration = 0;
async function api(path, options = {}) {
  const res = await fetch(path, {...options, headers:{'X-Admin-Request':'1', ...options.headers}});
  if (!res.ok) {let msg = `Ошибка ${res.status}`; try {msg = (await res.json()).error || msg;} catch {} throw Error(msg);}
  return res.json();
}
function element(tag, text, cls) {const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el;}
function button(text, cls, fn) {const el = element('button', text, cls); el.type = 'button'; el.onclick = async () => {el.disabled = true; try {await fn(); await refresh();} catch(e) {$('notice').textContent = e.message;} finally {el.disabled = false;}}; return el;}
$('add').onsubmit = async e => {e.preventDefault(); const btn = e.target.querySelector('button');btn.disabled = true;try {const r = await api('/api/jobs', {method:'POST',body:new FormData(e.target)}); selected = r.id; e.target.reset();$('notice').textContent = 'Задание сохранено. Можно закрыть браузер.';await refresh();} catch(e) {$('notice').textContent = e.message;} finally {btn.disabled = false;}};
async function refresh() {
  const generation = ++refreshGeneration;
  const jobs = await api('/api/jobs');
  if (generation !== refreshGeneration) return;
  if (selected && !jobs.some(job => job.id === selected)) selected = null;
  const list = document.createDocumentFragment();
  if (!jobs.length) list.append(element('p','Очередь пока пуста. Добавьте ссылку или TXT-файл выше.'));
  for (const job of jobs) {
    const card = element('article', undefined, 'job');
    const top = element('div', undefined, 'job-top');
    top.append(element('strong', `${job.kind === "rename" ? "Переименование" : job.urls.length+" ссылок"} · ${new Date(job.created*1000).toLocaleString()}`),element('span',(job.kind === "rename" && job.status === "running" ? "Переименовывается" : states[job.status]) || job.status,`state ${job.status}`));
    card.append(top,element('p',job.urls.slice(0,3).join('\n') + (job.urls.length > 3 ? '\n…' : '')));
    const actions = element('div',undefined,'actions');
    actions.append(button(selected === job.id ? 'Журнал открыт' : 'Открыть журнал','secondary',async()=>{selected=job.id;}));
    if (job.kind !== 'rename' && ['running','queued'].includes(job.status)) actions.append(button('Пауза','secondary',()=>api(`/api/jobs/${job.id}/pause`,{method:'POST'})));
    if (['running','queued','paused'].includes(job.status)) actions.append(button('Отменить','danger',()=>api(`/api/jobs/${job.id}/cancel`,{method:'POST'})));
    if (['paused','cancelled','failed'].includes(job.status)) actions.append(button('Продолжить','secondary',()=>api(`/api/jobs/${job.id}/resume`,{method:'POST'})));
    if (job.status === 'done') actions.append(button('Повторить','secondary',()=>api(`/api/jobs/${job.id}/retry`,{method:'POST'})));
    if (['queued','done','failed','cancelled','paused'].includes(job.status)) actions.append(button('Удалить','danger',async()=>{
      ++refreshGeneration;
      await api(`/api/jobs/${job.id}/delete`,{method:'POST'});
      if (selected === job.id) selected = null;
      $('notice').textContent = 'Задание и его журналы удалены. Скачанные файлы сохранены.';
    }));
    card.append(actions,element('p',`ID: ${job.id}${job.exit_code !== null ? ' · Код выхода: '+job.exit_code : ''}`,'meta'));list.append(card);
  }
  $('jobs').replaceChildren(list);
  if (!selected && jobs.length) selected = jobs.find(j=>j.status==='running')?.id || jobs[0].id;
  if (selected) {const data = await api(`/api/jobs/${selected}/log`);if (generation !== refreshGeneration) return;$('log-title').textContent = 'Журнал · '+selected.slice(0,8);$('log').textContent=data.log || 'Ожидание запуска…';if ($('follow').checked) $('log').scrollTop=$('log').scrollHeight;}
  else {$('log-title').textContent='Журнал задания';$('log').textContent='Выберите задание, чтобы открыть журнал.';}
  if (generation !== refreshGeneration) return;
  $('connection').textContent='Обновлено '+new Date().toLocaleTimeString();
  const runtime = await api('/api/runtime');$('runtime').textContent=runtime.message || '';
}
async function poll(){try {await refresh();} catch(e) {$('connection').textContent='Нет связи: '+e.message;} finally {setTimeout(poll,2000);}}
poll();

async function loadSettings() {
  try {const config=await api('/api/settings');$('metadata-enabled').checked=config.enabled;$('metadata-status').textContent=config.configured?'Токен сохранён. Настройки применятся при следующем запуске переименовщика.':'Токен ещё не задан. Сейчас доступен поиск Brave.';} catch(e) {$('metadata-status').textContent=e.message;}
}
$('settings').onsubmit=async e=>{e.preventDefault();const btn=e.target.querySelector('button');btn.disabled=true;try {await api('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:$('metadata-enabled').checked,tpdb_api_key:$('tpdb-key').value})});$('tpdb-key').value='';await loadSettings();} catch(e) {$('metadata-status').textContent=e.message;} finally {btn.disabled=false;}};
loadSettings();

$('rename').onclick=async()=>{const btn=$('rename');btn.disabled=true;try {const job=await api('/api/rename',{method:'POST'});selected=job.id;$('notice').textContent='Переименование добавлено в очередь.';await refresh();} catch(e) {$('notice').textContent=e.message;} finally {btn.disabled=false;}};

async function loadTelegram(){try{const config=await api('/api/telegram');$('telegram-enabled').checked=config.enabled;$('telegram-ids').value=config.allowed_user_ids.join(', ');$('telegram-status').textContent=(config.configured?'Токен сохранён. ':'Токен ещё не задан. ')+(config.enabled?'Бот включён; принимает сообщения только от разрешённых ID.':'Бот выключен.');}catch(e){$('telegram-status').textContent=e.message;}}
$('telegram-settings').onsubmit=async e=>{e.preventDefault();const btn=e.target.querySelector('button');btn.disabled=true;try{await api('/api/telegram',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:$('telegram-enabled').checked,token:$('telegram-token').value.trim(),allowed_user_ids:$('telegram-ids').value.split(/[\s,]+/).filter(Boolean)})});$('telegram-token').value='';await loadTelegram();}catch(e){$('telegram-status').textContent=e.message;}finally{btn.disabled=false;}};
loadTelegram();
