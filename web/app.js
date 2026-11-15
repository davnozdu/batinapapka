const $ = id => document.getElementById(id);
const states = {queued:'В очереди',running:'Скачивается',cancelling:'Отменяется',cancelled:'Отменено',done:'Завершено',failed:'Ошибка'};
let selected = null;
async function api(path, options = {}) {
  const res = await fetch(path, {...options, headers:{'X-Admin-Request':'1', ...options.headers}});
  if (!res.ok) {let msg = `Ошибка ${res.status}`; try {msg = (await res.json()).error || msg;} catch {} throw Error(msg);}
  return res.json();
}
function element(tag, text, cls) {const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el;}
function button(text, cls, fn) {const el = element('button', text, cls); el.type = 'button'; el.onclick = async () => {el.disabled = true; try {await fn(); await refresh();} catch(e) {$('notice').textContent = e.message;} finally {el.disabled = false;}}; return el;}
$('add').onsubmit = async e => {e.preventDefault(); const btn = e.target.querySelector('button');btn.disabled = true;try {const r = await api('/api/jobs', {method:'POST',body:new FormData(e.target)}); selected = r.id; e.target.reset();$('notice').textContent = 'Задание сохранено. Можно закрыть браузер.';await refresh();} catch(e) {$('notice').textContent = e.message;} finally {btn.disabled = false;}};
async function refresh() {
  const jobs = await api('/api/jobs');
  const list = document.createDocumentFragment();
  if (!jobs.length) list.append(element('p','Очередь пока пуста. Добавьте ссылку или TXT-файл выше.'));
  for (const job of jobs) {
    const card = element('article', undefined, 'job');
    const top = element('div', undefined, 'job-top');
    top.append(element('strong', `${job.urls.length} ссылок · ${new Date(job.created*1000).toLocaleString()}`),element('span',states[job.status] || job.status,`state ${job.status}`));
    card.append(top,element('p',job.urls.slice(0,3).join('\n') + (job.urls.length > 3 ? '\n…' : '')));
    const actions = element('div',undefined,'actions');
    actions.append(button(selected === job.id ? 'Журнал открыт' : 'Открыть журнал','secondary',async()=>{selected=job.id;}));
    if (['running','queued'].includes(job.status)) actions.append(button('Отменить','danger',()=>api(`/api/jobs/${job.id}/cancel`,{method:'POST'})));
    if (['done','failed','cancelled'].includes(job.status)) actions.append(button('Повторить','secondary',()=>api(`/api/jobs/${job.id}/retry`,{method:'POST'})));
    card.append(actions,element('p',`ID: ${job.id}${job.exit_code !== null ? ' · Код выхода: '+job.exit_code : ''}`,'meta'));list.append(card);
  }
  $('jobs').replaceChildren(list);
  if (!selected && jobs.length) selected = jobs.find(j=>j.status==='running')?.id || jobs[0].id;
  if (selected) {const data = await api(`/api/jobs/${selected}/log`);$('log-title').textContent = 'Журнал · '+selected.slice(0,8);$('log').textContent=data.log || 'Ожидание запуска…';if ($('follow').checked) $('log').scrollTop=$('log').scrollHeight;}
  $('connection').textContent='Обновлено '+new Date().toLocaleTimeString();
  const runtime = await api('/api/runtime');$('runtime').textContent=runtime.message || '';
}
async function poll(){try {await refresh();} catch(e) {$('connection').textContent='Нет связи: '+e.message;} finally {setTimeout(poll,2000);}}
poll();
